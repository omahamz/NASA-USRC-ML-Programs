using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

namespace SwGen
{
    /// <summary>
    /// A parametric SLDPRT whose geometry is driven by global-variable equations
    /// ("R" = 3.5, "A" = 60, ...). Wraps open/close, equation edits, rebuild checks
    /// and STEP export.
    /// </summary>
    internal sealed class PartTemplate : IDisposable
    {
        private const double MToMm = 1000.0;

        private readonly SldWorks app;
        private readonly ModelDoc2 doc;
        private readonly bool wasAlreadyOpen;
        private readonly Dictionary<int, string> originalEquations = new Dictionary<int, string>();

        public string PathName { get; }
        public string BaseName { get; }

        private PartTemplate(SldWorks app, ModelDoc2 doc, bool wasAlreadyOpen, string path)
        {
            this.app = app;
            this.doc = doc;
            this.wasAlreadyOpen = wasAlreadyOpen;
            PathName = path;
            BaseName = Path.GetFileNameWithoutExtension(path);
        }

        public static PartTemplate Open(SldWorks app, string partPath)
        {
            string full = Path.GetFullPath(partPath);
            if (!File.Exists(full)) throw new FileNotFoundException("Part file not found", full);

            // SolidWorks can't open two documents with the same file name, so reuse
            // the user's open copy if there is one (and restore its equations afterwards).
            var doc = (ModelDoc2)app.GetOpenDocumentByName(full);
            bool wasOpen = doc != null;
            if (!wasOpen)
            {
                int err = 0, warn = 0;
                doc = app.OpenDoc6(full, (int)swDocumentTypes_e.swDocPART,
                    (int)swOpenDocOptions_e.swOpenDocOptions_Silent, "", ref err, ref warn);
                if (doc == null)
                    throw new InvalidOperationException(
                        $"OpenDoc6 failed for {full} (swFileLoadError_e={err}, swFileLoadWarning_e={warn})");
            }
            if (doc.GetType() != (int)swDocumentTypes_e.swDocPART)
                throw new InvalidOperationException($"{full} is not a part document");

            Log.Info($"{(wasOpen ? "Using already-open" : "Opened")} part {full}");
            return new PartTemplate(app, doc, wasOpen, full);
        }

        // ── Equations ────────────────────────────────────────────────────────

        public JsonObject[] ListEquations()
        {
            var eq = doc.GetEquationMgr();
            int n = eq.GetCount();
            var list = new JsonObject[n];
            for (int k = 0; k < n; k++)
            {
                list[k] = new JsonObject
                {
                    ["index"] = k,
                    ["equation"] = eq.get_Equation(k),
                    ["value"] = eq.get_Value(k),
                    ["global_variable"] = eq.get_GlobalVariable(k),
                };
            }
            return list;
        }

        /// <summary>
        /// Index of the equation that *defines* <paramref name="name"/>, i.e. whose
        /// left-hand side is "name". The VBA macro matched "name" anywhere in the
        /// equation, which can hit a dimension equation that merely references it.
        /// </summary>
        public int FindDefiningEquation(string name)
        {
            var eq = doc.GetEquationMgr();
            string lhs = "\"" + name + "\"";
            for (int k = 0; k < eq.GetCount(); k++)
            {
                string expr = (eq.get_Equation(k) ?? "").TrimStart();
                if (expr.StartsWith(lhs, StringComparison.OrdinalIgnoreCase)
                    && expr.Substring(lhs.Length).TrimStart().StartsWith("="))
                    return k;
            }
            return -1;
        }

        // Literal right-hand side with an optional unit, then an optional 'comment:
        //   "A"= 90deg'Angle    "R"= 4mm'Circumradius    "CC"= 16'Circular Count
        private static readonly Regex LiteralEquation = new Regex(
            @"^\s*""[^""]+""\s*=\s*[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?\s*(?<unit>[A-Za-z]*)\s*(?<comment>'.*)?$",
            RegexOptions.Singleline);

        /// <summary>
        /// Replace the value of a global variable, keeping the template's unit suffix and
        /// comment. (The macro wrote a bare number, which dropped both and made the value
        /// depend on the document's unit settings.)
        /// </summary>
        public void SetEquation(int index, string name, double value)
        {
            var eq = doc.GetEquationMgr();
            if (!originalEquations.ContainsKey(index)) originalEquations[index] = eq.get_Equation(index);

            var m = LiteralEquation.Match(originalEquations[index]);
            string unit = m.Success ? m.Groups["unit"].Value : "";
            string comment = m.Success ? m.Groups["comment"].Value : "";
            eq.set_Equation(index, $"\"{name}\"= {value.ToString("R", CultureInfo.InvariantCulture)}{unit}{comment}");
        }

        public double GetEquationValue(int index) => doc.GetEquationMgr().get_Value(index);

        // ── Rebuild + checks ─────────────────────────────────────────────────

        public sealed class RebuildReport
        {
            public bool RebuildOk;
            public List<JsonObject> Errors = new List<JsonObject>();
            public List<JsonObject> Warnings = new List<JsonObject>();
        }

        public RebuildReport Rebuild()
        {
            var report = new RebuildReport();
            // EvaluateAll's return code is -1 for good and bad rows alike, so success is
            // judged by the rebuild result, What's Wrong list and value read-back instead.
            doc.GetEquationMgr().EvaluateAll();
            report.RebuildOk = doc.ForceRebuild3(false);

            var ext = doc.Extension;
            if (ext.GetWhatsWrongCount() > 0)
            {
                if (ext.GetWhatsWrong(out object feats, out object codes, out object warns))
                {
                    var f = (object[])feats;
                    var c = (int[])codes;
                    var w = (bool[])warns;
                    for (int i = 0; i < f.Length; i++)
                    {
                        var item = new JsonObject
                        {
                            ["feature"] = (f[i] as Feature)?.Name,
                            ["code"] = c[i],
                            ["code_name"] = Enum.GetName(typeof(swFeatureError_e), c[i]),
                        };
                        (w[i] ? report.Warnings : report.Errors).Add(item);
                    }
                }
            }
            return report;
        }

        public sealed class GeometryReport
        {
            public int SolidBodies;
            public int SheetBodies;
            public double VolumeMm3 = double.NaN;
            public double SurfaceAreaMm2 = double.NaN;
            public double[] BoundingBoxMm;   // xmin, ymin, zmin, xmax, ymax, zmax

            /// <summary>Fingerprint used to detect "rebuilt but geometry didn't change".</summary>
            public string Signature =>
                string.Join("|", new[] { VolumeMm3, SurfaceAreaMm2 }
                    .Concat(BoundingBoxMm ?? new double[0])
                    .Select(v => v.ToString("G10", CultureInfo.InvariantCulture)));
        }

        public GeometryReport MeasureGeometry()
        {
            var part = (PartDoc)doc;
            var g = new GeometryReport
            {
                SolidBodies = CountBodies(part, swBodyType_e.swSolidBody),
                SheetBodies = CountBodies(part, swBodyType_e.swSheetBody),
            };

            if (doc.Extension.CreateMassProperty2() is MassProperty2 mp)
            {
                g.VolumeMm3 = mp.Volume * Math.Pow(MToMm, 3);
                g.SurfaceAreaMm2 = mp.SurfaceArea * Math.Pow(MToMm, 2);
            }

            if (part.GetPartBox(true) is double[] box && box.Length == 6)
                g.BoundingBoxMm = box.Select(v => Math.Round(v * MToMm, 6)).ToArray();

            return g;
        }

        private static int CountBodies(PartDoc part, swBodyType_e type) =>
            part.GetBodies2((int)type, true) is object[] bodies ? bodies.Length : 0;

        // ── Export ───────────────────────────────────────────────────────────

        public sealed class ExportReport
        {
            public bool Ok;
            public int Errors;
            public int Warnings;
            public long Bytes;
            public string Message;
        }

        public ExportReport ExportStep(string stepPath)
        {
            var r = new ExportReport();
            // Delete first so "file exists afterwards" genuinely proves this export wrote it.
            if (File.Exists(stepPath)) File.Delete(stepPath);

            int errs = 0, warns = 0;
            bool saved = doc.Extension.SaveAs3(stepPath,
                (int)swSaveAsVersion_e.swSaveAsCurrentVersion,
                (int)(swSaveAsOptions_e.swSaveAsOptions_Silent | swSaveAsOptions_e.swSaveAsOptions_Copy),
                null, null, ref errs, ref warns);

            r.Errors = errs;
            r.Warnings = warns;
            var info = new FileInfo(stepPath);
            r.Bytes = info.Exists ? info.Length : 0;
            r.Ok = saved && errs == 0 && r.Bytes > 0;
            if (!r.Ok)
                r.Message = $"SaveAs3 returned {saved} (swFileSaveError_e={errs}, swFileSaveWarning_e={warns}, bytes={r.Bytes})";
            return r;
        }

        // ── Cleanup ──────────────────────────────────────────────────────────

        public void Dispose()
        {
            if (wasAlreadyOpen)
            {
                // Put the user's document back the way we found it.
                if (originalEquations.Count > 0)
                {
                    var eq = doc.GetEquationMgr();
                    foreach (var kv in originalEquations) eq.set_Equation(kv.Key, kv.Value);
                    eq.EvaluateAll();
                    doc.ForceRebuild3(false);
                    Log.Info($"Restored {originalEquations.Count} original equation(s) in the open document");
                }
            }
            else
            {
                // Closing discards the in-memory equation edits; the template on disk is never saved.
                app.CloseDoc(doc.GetTitle());
            }
        }
    }
}
