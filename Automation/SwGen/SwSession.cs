using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using SolidWorks.Interop.sldworks;

namespace SwGen
{
    /// <summary>
    /// Connection to a SolidWorks process. Attaches to a running instance when
    /// there is one (so the user's open session is reused), otherwise launches one.
    /// Only a SolidWorks that SwGen launched is shut down on Dispose.
    /// </summary>
    internal sealed class SwSession : IDisposable
    {
        private const string ProgId = "SldWorks.Application";
        private const string ProcessName = "SLDWORKS";

        private static readonly TimeSpan ExitTimeout = TimeSpan.FromSeconds(30);
        private static readonly TimeSpan StaleTimeout = TimeSpan.FromSeconds(30);

        public SldWorks App { get; }
        public bool Launched { get; }
        private readonly bool keepOpen;
        private readonly int processId;

        private SwSession(SldWorks app, bool launched, bool keepOpen, int processId)
        {
            App = app;
            Launched = launched;
            this.keepOpen = keepOpen;
            this.processId = processId;
        }

        public static SwSession Connect(bool visible, bool keepOpen)
        {
            try
            {
                var running = (SldWorks)Marshal.GetActiveObject(ProgId);
                Log.Info($"Attached to running SolidWorks {running.RevisionNumber()}");
                return new SwSession(running, launched: false, keepOpen, processId: 0);
            }
            catch (COMException)
            {
                // Not running (MK_E_UNAVAILABLE) — launch below.
            }

            WaitForUnreachableInstance();

            var type = Type.GetTypeFromProgID(ProgId)
                ?? throw new InvalidOperationException(
                    $"ProgID '{ProgId}' is not registered — is SolidWorks installed on this machine?");

            Log.Info("SolidWorks not running; launching (this can take a minute)...");
            var app = (SldWorks)Activator.CreateInstance(type);
            app.Visible = visible;
            app.UserControl = false;
            Log.Info($"Launched SolidWorks {app.RevisionNumber()}");
            return new SwSession(app, launched: true, keepOpen, ProcessIdOf(app));
        }

        /// <summary>
        /// A SolidWorks process that GetActiveObject cannot see is an instance that is starting, exiting, or was left
        /// behind by an earlier automation run. Activating a new instance would silently reconnect to it, and in that
        /// state equation edits have no effect (every row then fails the read-back check). Give it time to go away,
        /// then refuse instead of building on it.
        /// </summary>
        private static void WaitForUnreachableInstance()
        {
            var deadline = DateTime.UtcNow + StaleTimeout;
            bool announced = false;
            while (true)
            {
                var others = Process.GetProcessesByName(ProcessName);
                try
                {
                    if (others.Length == 0) return;
                    if (DateTime.UtcNow >= deadline)
                        throw new InvalidOperationException(
                            $"A SolidWorks process (pid {string.Join(", ", others.Select(p => p.Id))}) is running but SwGen cannot attach to it " +
                            "(usually left over from an earlier automation run). Close it or end it in Task Manager, then run SwGen again.");
                    if (!announced)
                        Log.Warn($"SolidWorks (pid {string.Join(", ", others.Select(p => p.Id))}) is running but not attachable; waiting up to {StaleTimeout.TotalSeconds:0}s for it to exit...");
                    announced = true;
                }
                finally
                {
                    foreach (var p in others) p.Dispose();
                }
                Thread.Sleep(1000);
            }
        }

        private static int ProcessIdOf(SldWorks app)
        {
            try { return app.GetProcessID(); }
            catch (COMException) { return 0; }
        }

        public void Dispose()
        {
            if (!Launched || keepOpen) return;
            try
            {
                App.ExitApp();
            }
            catch (COMException ex)
            {
                Log.Warn($"ExitApp failed: {ex.Message}");
            }

            // ExitApp alone is not enough: while this process holds COM references, an automation-launched SolidWorks
            // stays alive, and after we exit it lingers for minutes (until DCOM's ping timeout). Such a leftover is
            // invisible to GetActiveObject, so the next run would reconnect to it. Drop our references, then wait.
            ReleaseComReferences();
            WaitForProcessExit();
        }

        private void ReleaseComReferences()
        {
            try { Marshal.FinalReleaseComObject(App); }
            catch (ArgumentException) { /* not a COM object */ }
            catch (COMException) { /* already disconnected */ }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private void WaitForProcessExit()
        {
            const string closed = "Closed the SolidWorks instance SwGen launched";
            if (processId <= 0) { Log.Info(closed); return; }
            try
            {
                using (var p = Process.GetProcessById(processId))
                {
                    if (p.WaitForExit((int)ExitTimeout.TotalMilliseconds)) { Log.Info(closed); return; }
                    Log.Warn($"SolidWorks (pid {processId}) still running {ExitTimeout.TotalSeconds:0}s after ExitApp; terminating it");
                    p.Kill();
                    p.WaitForExit(5000);
                }
            }
            catch (ArgumentException) { Log.Info(closed); }             // no such process: already gone
            catch (InvalidOperationException ex) { Log.Warn($"Could not confirm SolidWorks exited: {ex.Message}"); }
            catch (Win32Exception ex) { Log.Warn($"Could not confirm SolidWorks exited: {ex.Message}"); }
        }
    }
}
