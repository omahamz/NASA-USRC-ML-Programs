# SwGen — SolidWorks STEP batch generator

C# console tool that drives SolidWorks through its COM API. It is a
programmatic replacement for the `Not Alternating(A).swp` batch-export macro:
for each CSV row it sets the part's global-variable equations, rebuilds,
**verifies the result**, and exports a `.stp`.

It is the build step of the agent pipeline (see
[docs/STP_AGENT_PIPELINE_PLAN.md](../../docs/STP_AGENT_PIPELINE_PLAN.md)). The
MCP server calls this executable.

## Requirements

- Windows with SolidWorks installed and licensed (tested with SolidWorks 2026, API 34.1)
- .NET SDK 8+ (for `dotnet build`) and the .NET Framework 4.8 targeting pack
  (installed with Visual Studio's ".NET desktop development" workload)

## Build

```bash
cd Automation/SwGen
dotnet build -c Release
# SolidWorks installed somewhere else:
dotnet build -c Release -p:SolidWorksApiDir="D:\SOLIDWORKS\api\redist"
```

The executable is written to `bin/Release/net48/SwGen.exe`.

PATH: C:\Users\User\Documents\Coding Stuff\SolidWorksStuff\3mmGap4mmThickN0\Shell

## Usage

```bash
SwGen probe                                      # connect and print the SolidWorks version
SwGen equations --part N6ASolid.SLDPRT           # dump equations / global variables as JSON
SwGen generate --part N6ASolid.SLDPRT --csv points.csv --out out_dir [--top 128] [--start 1]
               [--skip-columns P] [--prefix N6ASolid] [--step-ap 214] [--skip-existing]
```

```bash
bin/Release/net48/SwGen.exe probe
bin/Release/net48/SwGen.exe equations --part "C:\Users\User\Documents\Coding Stuff\SolidWorksStuff\3mmGap4mmThickN0\Shell\N0AShell.SLDPRT"
bin/Release/net48/SwGen.exe --part "C:\Users\User\Documents\Coding Stuff\SolidWorksStuff\3mmGap4mmThickN0\Shell\N0AShell.SLDPRT" --csv points.csv --out out_dir [--top 128] [--start 1] [--skip-columns P] [--prefix N6ASolid] [--step-ap 214] [--skip-existing]
```

- SwGen attaches to a running SolidWorks if there is one. Otherwise it launches
  SolidWorks and closes it when finished (`--keep-open` leaves it running).
- If the part is already open in SolidWorks, SwGen reuses that document and
  restores its original equations afterwards. The template file on disk is
  never saved.
- stdout carries JSON only; the progress log goes to stderr. Exit code: 0 = all
  rows ok, 1 = some failed or warned, 2 = fatal error.

Files written to `--out`:

- `<name>.stp`, named exactly as the macro named it (`N6ASolidR340A5861CC1800VC400`,
  which is what `src/analysis.py` parses)
- `swgen_results.jsonl`: one line per row with params, read-back values, rebuild
  errors, body count, volume, surface area, bounding box and status
- `swgen_summary.json`: counts and the list of problem rows

## What it checks that the macro didn't

| Check                                                                                                     | Macro behaviour                                                                            |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Every CSV column must have a _defining_ equation (`"R"= ...`), or the run stops before exporting anything | Printed a warning and exported anyway (the file name claims values the geometry never got) |
| Equation matched on the left-hand side only                                                               | Matched `"R"` anywhere, so it could hit `"D4@Sketch3"="R"`                                 |
| Unit suffix and comment kept (`"A"= 58.61deg'Angle`)                                                      | Wrote a bare `"A" = 58.61`, dropping both                                                  |
| Rebuild failure or What's Wrong errors → row **failed**, no STEP written                                  | Exported whatever geometry was left from the last good rebuild                             |
| Values read back from SolidWorks must match the request                                                   | —                                                                                          |
| At least one body is present; volume, area and bounding box recorded                                      | —                                                                                          |
| Identical geometry to the previous row despite different params → **warning**                             | —                                                                                          |
| Two rows that round to the same file name → second one **failed**                                         | Silently overwrote the first                                                               |
| Malformed or short CSV rows → **failed**                                                                  | Padded missing values with 0                                                               |
| STEP file must exist and be non-empty after export                                                        | Checked the return code only                                                               |

## One part per configuration (N, T)

The parts come in four families, `N6A…`, `N6TA…`, `N0A…`, `N0TA…` (`…Shell.SLDPRT` or `…Solid.SLDPRT`),
selected by the `N` (6 = hexagonal cells, 0 = ellipsoidal holes) and `T` (1 = twist) inputs of the ML model.
SwGen builds **one part per run** and binds every CSV column to a global variable of that part, so:

- `T` and `N` are not equations. A CSV that still carries them aborts with "no defining equation" unless you
  pass `--skip-columns P,T,N`.
- `python src/stp_preflight.py points.csv --OD 40 --L 50 --G 3 --fidelity Shell --out batches/run1` checks
  every point against the constraints of its own configuration (`src/constraints.py`), rejects infeasible,
  out-of-range, non-integer `CC`/`VC` and file-name-colliding points, writes one `R,A,CC,VC` CSV per
  configuration and prints the matching `SwGen generate` command for each.
- The STEP file name is `<part name><R><A><CC><VC>` with every value written as `round(value * 100)`
  (half to even), e.g. `N0AShellR510A6000CC1200VC500`; the angle label is `A` for twisted parts too.
- SwGen shuts down a SolidWorks it launched by releasing its COM references after `ExitApp()` and waiting for the
  process to exit (ending it after 30 s). Without that, the headless instance lingers for minutes; it is invisible to
  `GetActiveObject`, so the next run silently reconnected to it and every equation edit was a no-op (rows then failed
  the read-back check). On start SwGen now waits for such an unreachable `SLDWORKS.exe` to exit and refuses to run
  against it.
- `python -m pytest -m solidworks` runs the opt-in SwGen tests (needs SolidWorks; set `SWGEN_PARTS_DIR` to the
  folder holding the SLDPRT files).
