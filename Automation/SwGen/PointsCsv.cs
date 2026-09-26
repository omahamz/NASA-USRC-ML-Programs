using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SwGen
{
    /// <summary>One design point: raw text values keyed by CSV header, in header order.</summary>
    internal sealed class DesignPoint
    {
        public int Row { get; }                 // 1-based data row (header excluded)
        public IReadOnlyList<string> Values { get; }
        public string Error { get; }            // non-null if the row is malformed

        public DesignPoint(int row, IReadOnlyList<string> values, string error)
        {
            Row = row;
            Values = values;
            Error = error;
        }
    }

    internal sealed class PointsCsv
    {
        public IReadOnlyList<string> Headers { get; }
        public IReadOnlyList<DesignPoint> Points { get; }

        private PointsCsv(IReadOnlyList<string> headers, IReadOnlyList<DesignPoint> points)
        {
            Headers = headers;
            Points = points;
        }

        /// <summary>
        /// Plain comma-separated file, first line = headers (same format the VBA macro read).
        /// Unlike the macro, a short row is reported as an error rather than padded with 0,
        /// so a truncated line can never silently become a zero-valued design.
        /// </summary>
        public static PointsCsv Load(string path)
        {
            var lines = File.ReadAllLines(path).Where(l => l.Trim().Length > 0).ToList();
            if (lines.Count < 2) throw new InvalidDataException($"{path}: needs a header row and at least one data row");

            var headers = lines[0].Split(',').Select(h => h.Trim().TrimStart('﻿')).ToList();
            var points = new List<DesignPoint>();

            for (int i = 1; i < lines.Count; i++)
            {
                var cols = lines[i].Split(',').Select(c => c.Trim()).ToList();
                string error = cols.Count == headers.Count
                    ? null
                    : $"expected {headers.Count} columns, found {cols.Count}";
                points.Add(new DesignPoint(i, cols, error));
            }

            var dup = headers.GroupBy(h => h, StringComparer.OrdinalIgnoreCase).FirstOrDefault(g => g.Count() > 1);
            if (dup != null) throw new InvalidDataException($"{path}: duplicate column '{dup.Key}'");

            return new PointsCsv(headers, points);
        }
    }
}
