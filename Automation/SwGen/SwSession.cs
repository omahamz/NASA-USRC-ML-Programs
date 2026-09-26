using System;
using System.Runtime.InteropServices;
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

        public SldWorks App { get; }
        public bool Launched { get; }
        private readonly bool keepOpen;

        private SwSession(SldWorks app, bool launched, bool keepOpen)
        {
            App = app;
            Launched = launched;
            this.keepOpen = keepOpen;
        }

        public static SwSession Connect(bool visible, bool keepOpen)
        {
            try
            {
                var running = (SldWorks)Marshal.GetActiveObject(ProgId);
                Log.Info($"Attached to running SolidWorks {running.RevisionNumber()}");
                return new SwSession(running, launched: false, keepOpen);
            }
            catch (COMException)
            {
                // Not running (MK_E_UNAVAILABLE) — launch below.
            }

            var type = Type.GetTypeFromProgID(ProgId)
                ?? throw new InvalidOperationException(
                    $"ProgID '{ProgId}' is not registered — is SolidWorks installed on this machine?");

            Log.Info("SolidWorks not running; launching (this can take a minute)...");
            var app = (SldWorks)Activator.CreateInstance(type);
            app.Visible = visible;
            app.UserControl = false;
            Log.Info($"Launched SolidWorks {app.RevisionNumber()}");
            return new SwSession(app, launched: true, keepOpen);
        }

        public void Dispose()
        {
            if (!Launched || keepOpen) return;
            try
            {
                App.ExitApp();
                Log.Info("Closed the SolidWorks instance SwGen launched");
            }
            catch (COMException ex)
            {
                Log.Warn($"ExitApp failed: {ex.Message}");
            }
        }
    }
}
