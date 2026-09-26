using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace SwGen
{
    /// <summary>Progress log. Goes to stderr so stdout stays pure JSON for callers.</summary>
    internal static class Log
    {
        public static void Info(string msg) => Write("INFO ", msg);
        public static void Warn(string msg) => Write("WARN ", msg);
        public static void Error(string msg) => Write("ERROR", msg);

        private static void Write(string level, string msg) =>
            Console.Error.WriteLine($"{DateTime.Now:HH:mm:ss} {level} {msg}");
    }

    /// <summary>Minimal "--key value" / "--flag" parser.</summary>
    internal sealed class Options
    {
        private static readonly HashSet<string> Flags = new HashSet<string>
        {
            "visible", "keep-open", "skip-existing",
        };

        private readonly Dictionary<string, string> values = new Dictionary<string, string>();

        public static Options Parse(string[] args, int start)
        {
            var o = new Options();
            for (int i = start; i < args.Length; i++)
            {
                string a = args[i];
                if (!a.StartsWith("--")) throw new ArgumentException($"unexpected argument '{a}'");
                string key = a.Substring(2);
                if (Flags.Contains(key)) { o.values[key] = "true"; continue; }
                if (i + 1 >= args.Length) throw new ArgumentException($"--{key} needs a value");
                o.values[key] = args[++i];
            }
            return o;
        }

        public bool Flag(string key) => values.ContainsKey(key);

        public string Get(string key, string fallback = null) =>
            values.TryGetValue(key, out var v) ? v : fallback;

        public string Require(string key) =>
            Get(key) ?? throw new ArgumentException($"--{key} is required");

        public int GetInt(string key, int fallback)
        {
            string v = Get(key);
            if (v == null) return fallback;
            if (!int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n))
                throw new ArgumentException($"--{key} must be an integer, got '{v}'");
            return n;
        }
    }

    /// <summary>Insertion-ordered JSON object.</summary>
    internal sealed class JsonObject : IEnumerable<KeyValuePair<string, object>>
    {
        private readonly List<KeyValuePair<string, object>> items = new List<KeyValuePair<string, object>>();

        public object this[string key]
        {
            get => items.FirstOrDefault(p => p.Key == key).Value;
            set
            {
                int i = items.FindIndex(p => p.Key == key);
                var pair = new KeyValuePair<string, object>(key, value);
                if (i >= 0) items[i] = pair; else items.Add(pair);
            }
        }

        public IEnumerator<KeyValuePair<string, object>> GetEnumerator() => items.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    /// <summary>Tiny JSON serializer (net48 has no System.Text.Json; avoids a NuGet dependency).</summary>
    internal static class Json
    {
        public static string Serialize(object value)
        {
            var sb = new StringBuilder();
            Write(sb, value);
            return sb.ToString();
        }

        private static void Write(StringBuilder sb, object v)
        {
            switch (v)
            {
                case null: sb.Append("null"); break;
                case string s: WriteString(sb, s); break;
                case bool b: sb.Append(b ? "true" : "false"); break;
                case double d:
                    sb.Append(double.IsNaN(d) || double.IsInfinity(d) ? "null" : d.ToString("R", CultureInfo.InvariantCulture));
                    break;
                case float f: Write(sb, (double)f); break;
                case int or long or short: sb.Append(Convert.ToString(v, CultureInfo.InvariantCulture)); break;
                case JsonObject obj:
                    sb.Append('{');
                    bool first = true;
                    foreach (var kv in obj)
                    {
                        if (!first) sb.Append(',');
                        first = false;
                        WriteString(sb, kv.Key);
                        sb.Append(':');
                        Write(sb, kv.Value);
                    }
                    sb.Append('}');
                    break;
                case IEnumerable seq:
                    sb.Append('[');
                    bool firstItem = true;
                    foreach (var item in seq)
                    {
                        if (!firstItem) sb.Append(',');
                        firstItem = false;
                        Write(sb, item);
                    }
                    sb.Append(']');
                    break;
                default: WriteString(sb, v.ToString()); break;
            }
        }

        private static void WriteString(StringBuilder sb, string s)
        {
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4"));
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }
    }
}
