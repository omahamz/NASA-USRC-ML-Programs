using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using SolidWorks.Interop.swconst;

namespace SwGen
{
    internal static class Commands
    {
        // ── probe ────────────────────────────────────────────────────────────

        public static int Probe(Options o)
        {
            using var session = SwSession.Connect(o.Flag("visible"), o.Flag("keep-open"));
            var app = session.App;
            Console.WriteLine(Json.Serialize(new JsonObject
            {
                ["ok"] = true,
                ["revision"] = app.RevisionNumber(),
                ["launched_by_swgen"] = session.Launched,
                ["visible"] = app.Visible,
                ["open_documents"] = app.GetDocumentCount(),
            }));
            return 0;
        }

        // ── equations ────────────────────────────────────────────────────────

        public static int Equations(Options o)
        {
            string part = o.Require("part");
            using var session = SwSession.Connect(o.Flag("visible"), o.Flag("keep-open"));
            using var template = PartTemplate.Open(session.App, part);
            Console.WriteLine(Json.Serialize(new JsonObject
            {
                ["part"] = template.PathName,
                ["equations"] = template.ListEquations(),
            }));
            return 0;
        }

        // ── generate ─────────────────────────────────────────────────────────

        /// <summary>A CSV column bound to the part equation it drives.</summary>
        private sealed class Binding
        {
            public int Column;          // index into the CSV row
            public string Header;       // CSV header, e.g. "A"
            public string Equation;     // global variable name in the part, e.g. "TA"
            public int EquationIndex;
            public string FileLabel;    // label used in the output file name
        }

        public static int Generate(Options o)
        {
            string partPath = o.Require("part");
            string csvPath = o.Require("csv");
            string outDir = Path.GetFullPath(o.Require("out"));
            int start = o.GetInt("start", 1);
            int top = o.GetInt("top", -1);
            bool skipExisting = o.Flag("skip-existing");
            int stepAp = o.GetInt("step-ap", 0);
            var skipColumns = new HashSet<string>(
                o.Get("skip-columns", "P").Split(',').Select(s => s.Trim()).Where(s => s.Length > 0),
                StringComparer.OrdinalIgnoreCase);

            if (start < 1) throw new ArgumentException("--start must be >= 1");
            if (stepAp != 0 && stepAp != 203 && stepAp != 214) throw new ArgumentException("--step-ap must be 203 or 214");

            var csv = PointsCsv.Load(csvPath);
            Directory.CreateDirectory(outDir);

            var total = Stopwatch.StartNew();
            using var session = SwSession.Connect(o.Flag("visible"), o.Flag("keep-open"));
            var app = session.App;
            using var template = PartTemplate.Open(app, partPath);

            string prefix = o.Get("prefix", template.BaseName);
            var bindings = BindColumns(csv.Headers, skipColumns, template);
            Log.Info("Bindings: " + string.Join(", ", bindings.Select(b =>
                b.Header == b.Equation ? b.Header : $"{b.Header}->\"{b.Equation}\"")));

            var rows = csv.Points.Skip(start - 1);
            if (top >= 0) rows = rows.Take(top);
            var selected = rows.ToList();

            int originalStepAp = app.GetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swStepAP);
            if (stepAp != 0) app.SetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swStepAP, stepAp);

            var counts = new Dictionary<string, int> { ["ok"] = 0, ["warning"] = 0, ["failed"] = 0, ["skipped"] = 0 };
            var problemRows = new List<JsonObject>();
            string resultsPath = Path.Combine(outDir, "swgen_results.jsonl");

            try
            {
                // Suppresses UI updates between API calls; noticeably faster for batches.
                app.CommandInProgress = true;

                using var results = new StreamWriter(resultsPath, append: false);
                var usedNames = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                string prevSignature = null;
                double[] prevValues = null;

                for (int i = 0; i < selected.Count; i++)
                {
                    var point = selected[i];
                    var result = GenerateRow(point, bindings, template, prefix, outDir, skipExisting,
                        usedNames, ref prevSignature, ref prevValues);

                    string status = (string)result["status"];
                    counts[status]++;
                    if (status == "failed" || status == "warning")
                        problemRows.Add(new JsonObject
                        {
                            ["row"] = point.Row,
                            ["status"] = status,
                            ["reasons"] = result[status == "failed" ? "errors" : "warnings"],
                        });

                    results.WriteLine(Json.Serialize(result));
                    results.Flush(); // keep partial results if SolidWorks dies mid-batch
                    string name = (string)result["name"] ?? "";
                    Log.Info($"[{i + 1}/{selected.Count}] row {point.Row} {status.ToUpperInvariant()} {name}");
                }
            }
            finally
            {
                app.CommandInProgress = false;
                if (stepAp != 0) app.SetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swStepAP, originalStepAp);
            }

            var summary = new JsonObject
            {
                ["part"] = template.PathName,
                ["csv"] = Path.GetFullPath(csvPath),
                ["out"] = outDir,
                ["solidworks"] = app.RevisionNumber(),
                ["step_ap"] = stepAp != 0 ? stepAp : originalStepAp,
                ["rows_in_csv"] = csv.Points.Count,
                ["processed"] = selected.Count,
                ["ok"] = counts["ok"],
                ["warning"] = counts["warning"],
                ["failed"] = counts["failed"],
                ["skipped"] = counts["skipped"],
                ["problems"] = problemRows,
                ["results_file"] = resultsPath,
                ["elapsed_s"] = Math.Round(total.Elapsed.TotalSeconds, 2),
            };
            string summaryJson = Json.Serialize(summary);
            File.WriteAllText(Path.Combine(outDir, "swgen_summary.json"), summaryJson);
            Console.WriteLine(summaryJson);

            Log.Info($"Done: {counts["ok"]} ok, {counts["warning"]} warning, {counts["failed"]} failed, {counts["skipped"]} skipped");
            return counts["failed"] + counts["warning"] == 0 ? 0 : 1;
        }

        /// <summary>
        /// Map each CSV column to the global variable it drives. A column with no
        /// matching equation is a hard error up front: the VBA macro only printed a
        /// warning and went on exporting, producing files whose name claims a value
        /// the geometry never received.
        /// </summary>
        private static List<Binding> BindColumns(IReadOnlyList<string> headers, HashSet<string> skip, PartTemplate template)
        {
            // Same rule as the macro. Downstream parsers (src/analysis.py) expect the
            // angle labelled "A" in file names; the TA/A distinction lives in the prefix.
            string angleLabel = template.BaseName.Equals("N600TA", StringComparison.OrdinalIgnoreCase) ? "TA" : "A";

            var bindings = new List<Binding>();
            var missing = new List<string>();
            for (int c = 0; c < headers.Count; c++)
            {
                string h = headers[c];
                if (skip.Contains(h)) continue;

                string eqName = h;
                int idx = template.FindDefiningEquation(h);
                string alt = AlternateName(h);
                if (idx < 0 && alt != null)
                {
                    idx = template.FindDefiningEquation(alt);
                    eqName = alt;
                }
                if (idx < 0) { missing.Add(h); continue; }

                bindings.Add(new Binding
                {
                    Column = c,
                    Header = h,
                    Equation = eqName,
                    EquationIndex = idx,
                    FileLabel = h is "A" or "TA" ? angleLabel : h,
                });
            }

            if (missing.Count > 0)
            {
                var available = template.ListEquations().Select(e => (string)e["equation"]);
                throw new InvalidOperationException(
                    $"CSV column(s) {string.Join(", ", missing)} have no defining equation in {template.PathName}. " +
                    $"Add them to --skip-columns if they are not geometry inputs. Part equations: {string.Join(" | ", available)}");
            }
            return bindings;
        }

        private static string AlternateName(string name) => name switch
        {
            "A" => "TA",
            "TA" => "A",
            _ => null,
        };

        /// <summary>Macro-compatible name segment: label + round(value * 100), banker's rounding like VBA Round.</summary>
        private static string FileSegment(string label, double value) =>
            label + Math.Round(value * 100, MidpointRounding.ToEven).ToString("0", CultureInfo.InvariantCulture);

        private static JsonObject GenerateRow(
            DesignPoint point, List<Binding> bindings, PartTemplate template, string prefix, string outDir,
            bool skipExisting, Dictionary<string, int> usedNames, ref string prevSignature, ref double[] prevValues)
        {
            var timer = Stopwatch.StartNew();
            var errors = new List<string>();
            var warnings = new List<string>();
            var result = new JsonObject { ["row"] = point.Row };

            JsonObject Finish(string status)
            {
                result["status"] = status;
                result["errors"] = errors;
                result["warnings"] = warnings;
                result["elapsed_s"] = Math.Round(timer.Elapsed.TotalSeconds, 2);
                return result;
            }

            if (point.Error != null)
            {
                errors.Add("malformed CSV row: " + point.Error);
                return Finish("failed");
            }

            // Parse the values this row drives.
            var values = new double[bindings.Count];
            var parameters = new JsonObject();
            for (int b = 0; b < bindings.Count; b++)
            {
                string text = point.Values[bindings[b].Column];
                if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out values[b]))
                    errors.Add($"{bindings[b].Header}='{text}' is not a number");
                parameters[bindings[b].Header] = values[b];
            }
            result["params"] = parameters;
            if (errors.Count > 0) return Finish("failed");

            string name = prefix + string.Concat(bindings.Select((b, k) => FileSegment(b.FileLabel, values[k])));
            string stepPath = Path.Combine(outDir, name + ".stp");
            result["name"] = name;

            // Two rows that round to the same file name would silently overwrite each other.
            if (usedNames.TryGetValue(name, out int firstRow))
            {
                errors.Add($"file name {name} collides with row {firstRow} (values equal after rounding to 0.01)");
                return Finish("failed");
            }
            usedNames[name] = point.Row;

            if (skipExisting && File.Exists(stepPath))
            {
                result["step_path"] = stepPath;
                return Finish("skipped");
            }

            try
            {
                for (int b = 0; b < bindings.Count; b++)
                    template.SetEquation(bindings[b].EquationIndex, bindings[b].Equation, values[b]);

                var rebuild = template.Rebuild();
                result["rebuild_errors"] = rebuild.Errors;
                result["rebuild_warnings"] = rebuild.Warnings;
                if (!rebuild.RebuildOk) errors.Add("ForceRebuild3 returned false");
                foreach (var e in rebuild.Errors) errors.Add("rebuild error: " + Json.Serialize(e));
                foreach (var w in rebuild.Warnings) warnings.Add("rebuild warning: " + Json.Serialize(w));

                // Read back: proves SolidWorks actually holds the values we asked for.
                var readback = new JsonObject();
                for (int b = 0; b < bindings.Count; b++)
                {
                    double actual = template.GetEquationValue(bindings[b].EquationIndex);
                    readback[bindings[b].Equation] = actual;
                    if (Math.Abs(actual - values[b]) > 1e-9 * Math.Max(1.0, Math.Abs(values[b])))
                        errors.Add($"\"{bindings[b].Equation}\" reads back {actual}, expected {values[b]}");
                }
                result["readback"] = readback;

                var geom = template.MeasureGeometry();
                result["solid_bodies"] = geom.SolidBodies;
                result["sheet_bodies"] = geom.SheetBodies;
                result["volume_mm3"] = geom.VolumeMm3;
                result["surface_area_mm2"] = geom.SurfaceAreaMm2;
                result["bbox_mm"] = geom.BoundingBoxMm;
                if (geom.SolidBodies + geom.SheetBodies == 0) errors.Add("part has no bodies after rebuild");

                // Different inputs but identical geometry = SolidWorks ignored the change.
                string signature = geom.Signature;
                if (prevSignature != null && signature == prevSignature && !values.SequenceEqual(prevValues))
                    warnings.Add("geometry identical to previous row despite different parameters (stale rebuild?)");
                prevSignature = signature;
                prevValues = values;

                if (errors.Count > 0) return Finish("failed"); // never export a part we know is wrong

                var export = template.ExportStep(stepPath);
                result["step_path"] = stepPath;
                result["step_bytes"] = export.Bytes;
                if (!export.Ok) errors.Add(export.Message);
                else if (export.Warnings != 0) warnings.Add($"SaveAs3 warning code {export.Warnings}");
            }
            catch (COMException ex) when (!IsServerGone(ex))
            {
                errors.Add($"COM error 0x{ex.HResult:X8}: {ex.Message}");
            }

            return Finish(errors.Count > 0 ? "failed" : warnings.Count > 0 ? "warning" : "ok");
        }

        /// <summary>SolidWorks crashed or was closed — every later call would fail, so abort the batch.</summary>
        private static bool IsServerGone(COMException ex) =>
            (uint)ex.HResult is 0x800706BA   // RPC_S_SERVER_UNAVAILABLE
                             or 0x80010108   // RPC_E_DISCONNECTED
                             or 0x800706BE;  // RPC_S_CALL_FAILED
    }
}
