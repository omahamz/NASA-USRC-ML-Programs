using System;
using System.Globalization;
using System.Threading;

namespace SwGen
{
    internal static class Program
    {
        private const string Usage = @"SwGen — SolidWorks STEP batch generator

Usage:
  SwGen probe [--visible]
      Connect to SolidWorks (attach, or launch if not running) and report its version.

  SwGen equations --part <file.SLDPRT>
      List the part's equations / global variables as JSON.

  SwGen generate --part <file.SLDPRT> --csv <points.csv> --out <dir> [options]
      For each CSV row: set the matching global variables, rebuild, verify, export .stp.
      Options:
        --start <n>          First data row to process, 1-based (default 1)
        --top <n>            Max rows to process, -1 = all (default -1)
        --skip-columns <a,b> CSV columns that are not equations (default P)
        --prefix <name>      File name prefix (default: part file name)
        --step-ap <203|214>  Force STEP application protocol for this run (restored after)
        --skip-existing      Skip rows whose .stp already exists in --out
        --visible            Show the SolidWorks window if SwGen has to launch it
        --keep-open          Leave SolidWorks running if SwGen launched it

Output:
  stdout  JSON (probe / equation list / batch summary) — machine readable
  stderr  progress log
  --out   <name>.stp files, swgen_results.jsonl (one line per row), swgen_summary.json

Exit codes: 0 = all rows ok, 1 = some rows failed or warned, 2 = fatal / bad arguments";

        [STAThread] // SolidWorks COM objects live in a single-threaded apartment
        private static int Main(string[] args)
        {
            // Equation text and file names must never pick up a locale decimal comma.
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
            {
                Console.Error.WriteLine(Usage);
                return args.Length == 0 ? 2 : 0;
            }

            try
            {
                var opts = Options.Parse(args, 1);
                switch (args[0])
                {
                    case "probe": return Commands.Probe(opts);
                    case "equations": return Commands.Equations(opts);
                    case "generate": return Commands.Generate(opts);
                    default: throw new ArgumentException($"unknown command '{args[0]}'");
                }
            }
            catch (ArgumentException ex)
            {
                Console.Error.WriteLine("error: " + ex.Message);
                Console.Error.WriteLine();
                Console.Error.WriteLine(Usage);
                return 2;
            }
            catch (Exception ex)
            {
                Log.Error(ex.ToString());
                Console.WriteLine(Json.Serialize(new JsonObject { ["fatal"] = ex.Message }));
                return 2;
            }
        }
    }
}
