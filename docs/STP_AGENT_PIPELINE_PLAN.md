# Plan: Agent-orchestrated STEP generation (SolidWorks C# API + MCP)

## Context
The end goal is a closed loop: ML surrogate proposes a batch → SolidWorks builds parts → agent validates STEPs → Abaqus simulates (`Automation/CompTest_PipelineParallel.py`) → agent reviews results → ML fine-tune. This plan covers **only stage 1**: letting an LLM agent (Gemini for now) orchestrate STEP generation with zero silent build failures. Two pieces:
1. Drive SolidWorks programmatically from this repo in C# (replacing the in-app VBA macro that edits a template part).
2. Build an MCP server exposing CAD + validation tools, and connect a Gemini-powered agent to it.

Setup facts: SolidWorks and Abaqus are on **different machines**; the macro **edits a template .SLDPRT**; the repo's data is confidential (see `.gitignore`: data_folder/, other/, models/).

---

## Key concept to get straight first
- **The API key belongs to the agent, not the MCP server.** The agent (the "MCP client/host") holds `GEMINI_API_KEY` and calls Gemini. The MCP server is just a local program exposing tools; it never sees the Gemini key.
- If the MCP server is ever exposed over the network (HTTP), protect it with its **own** bearer token — separate from Gemini.
- **Yes, write your own tools.** Nothing off-the-shelf knows your template, constraints, or pass/fail criteria. Rule: *tools are deterministic code that decide pass/fail; the agent only decides what to do next* (retry, perturb, flag, report). Never let the LLM "eyeball" whether a part is valid.

---

## Part 1 — Connecting SolidWorks to C#

### How it works
SolidWorks exposes a **COM API**. Any .NET program on the SolidWorks machine can attach to it — the same object model your VBA macros use (`SldWorks`, `ModelDoc2`, `PartDoc`…), so recorded macro code translates almost line-for-line.

### Steps
1. **Prereqs (SolidWorks machine):** SolidWorks installed + licensed (confirm your edition — Education/Research licences allow API use; verify if on Student edition). Install Visual Studio 2022 (".NET desktop development") or the .NET SDK + VS Code. Optionally install the *SOLIDWORKS API SDK* from the SW installer (templates + local API help).
2. **Create the project:** `Automation/SwGen/` — a C# console app targeting **.NET Framework 4.8** (simplest COM interop; `net8.0-windows` also works but `Marshal.GetActiveObject` is missing there).
3. **Reference the interop DLLs** from `C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS\api\redist\`:
   `SolidWorks.Interop.sldworks.dll`, `SolidWorks.Interop.swconst.dll` (set *Embed Interop Types = False*).
4. **Attach/launch:**
   ```csharp
   var t  = Type.GetTypeFromProgID("SldWorks.Application");
   var sw = (SldWorks)Activator.CreateInstance(t);   // launches or attaches
   sw.Visible = false; sw.UserControl = false;         // headless-ish, no user dialogs
   ```
   Mark `Main` with `[STAThread]` (COM is single-threaded apartment).
5. **Port the macro logic** (open template → set params → rebuild → export):
   - `OpenDoc6(templatePath, swDocPART, swOpenDocOptions_Silent, ...)`
   - Set values via dimensions (`doc.Parameter("D1@Sketch1").SystemValue = …`) or global variables (`doc.GetEquationMgr()`) — whichever the template uses. **API units are meters and radians** (classic silent bug).
   - `doc.ForceRebuild3(false)`, then collect errors: `doc.Extension.GetWhatsWrongCount()` / `GetWhatsWrong(...)`, and per-feature `GetErrorCode2`.
   - Export: set STEP AP214 via `SetUserPreferenceIntegerValue(swStepAP, 214)`, then `doc.Extension.SaveAs3(path.step, ..., swSaveAsOptions_Silent, ...)` and check `errors/warnings`.
   - Capture self-check data **from SolidWorks** before closing: mass properties (`Extension.CreateMassProperty2` → volume, surface area), body count (`GetBodies2`), bounding box, and **read back** every parameter to confirm it actually changed.
   - `sw.CloseDoc(...)` — never save over the template (open read-only or `SaveAs` copies only).
6. **Package as a CLI** (the only thing the MCP server calls):
   ```
   SwGen.exe generate --batch batches/<id>/manifest.json --out batches/<id>/step
   SwGen.exe probe                       # connectivity/version check
   ```
   Output: per-point JSON result `{id, status, rebuild_errors[], params_readback{}, volume, bbox, body_count, step_path, elapsed_s}`.
7. **Robustness:** process points serially (one SW instance; parallel SW is unreliable), per-part timeout, on hang/COM exception kill `sldworks.exe` and restart, skip points already `ok` (resumable), write results incrementally.

### Files
- `Automation/SwGen/` — `SwGen.csproj`, `Program.cs` (CLI), `SwSession.cs` (attach/restart/timeouts), `PartBuilder.cs` (param set + rebuild + checks), `StepExporter.cs`, `param_map.json` (ML variable → SW dimension/global-var name, unit, conversion).
- Replace the empty `Automation/test.cs`.

---

## Part 2 — MCP server + agent

### Architecture
```
[Gemini agent (python, holds GEMINI_API_KEY)]
        │ MCP (stdio if same machine; streamable HTTP + bearer token if remote)
[MCP server: Python FastMCP, runs on SolidWorks PC]
        ├─ calls SwGen.exe (subprocess)
        ├─ imports src/check_constraints.py (reuse violates_c1 / violates_c2)
        └─ STEP validation via cadquery/OCP (independent of SolidWorks)
```
Python server (not C#) because the rest of the repo — constraints, sampling, ML — is Python and can be imported directly. (C# MCP SDK exists if you later want to merge.)

### Tools to implement (`Automation/mcp_server/`)
| Tool | Purpose |
|---|---|
| `check_constraints(points)` | Reject infeasible points before CAD; wraps `src/check_constraints.py` |
| `create_batch(points, source)` | Write `batches/<id>/manifest.json` (ids, params, provenance e.g. model version) |
| `generate_parts(batch_id, ids?)` | Run `SwGen.exe`; return per-point status |
| `get_build_log(batch_id, id)` | Rebuild errors / What's Wrong details for triage |
| `validate_step(batch_id, id)` | Open STEP independently: 1 solid, closed/valid shape, volume & bbox vs SW mass props + expected OD/L, params readback match |
| `batch_status(batch_id)` | Counts ok / failed / invalid / pending |
| `retry_point(batch_id, id, adjusted_params?)` | Regenerate one point (agent may nudge params within bounds; logged) |
| `finalize_batch(batch_id)` | Write `report.json` + zip of validated STEPs ready to ship to Abaqus machine |

Also expose read-only **resources**: design-space bounds, `param_map.json`, pass/fail tolerances — so the agent reasons with the same rules the code enforces.

### Agent (`Automation/agent/run_agent.py`)
- `google-genai` SDK + `mcp` Python client; Gemini's SDK can take an MCP `ClientSession` directly as a tool (automatic function calling). Fallback: manual loop mapping MCP tool schemas → Gemini function declarations (keeps it provider-agnostic for a later Claude/OpenAI swap).
- Quick alternative for early testing: **Gemini CLI** with the server registered in its `settings.json` `mcpServers` block — zero agent code.
- System prompt: goal = "every point in the batch is either validated or explicitly rejected with a reason"; max retries per point; never modify points outside bounds; stop and report rather than guess.
- Key from env var / `.env` (add `.env` to `.gitignore`).

### Files
- `Automation/mcp_server/server.py`, `tools/{constraints.py, cad.py, step_validate.py, batch.py}`, `config.yaml` (paths to SwGen.exe, template, tolerances)
- `Automation/agent/run_agent.py`, `prompts/system.md`
- `batches/` (gitignored), `requirements.txt` additions: `mcp`, `google-genai`, `cadquery` (or `cadquery-ocp`), `python-dotenv`

---

## Things you may have missed
1. **Silent wrong geometry is the real risk, not crashes.** SolidWorks can "rebuild OK" but ignore a dimension that over-constrains a sketch, exporting the *previous* shape. Hence read-back + independent volume/bbox checks.
2. **Units** — API is meters/radians; your ML params are mm/degrees.
3. **Parameter mapping** — ML inputs are `R, A, CC, VC`, but sample files (`src/src_data/test_points.csv`) also carry `ID`, `P`; the template may need OD/L/G too. Pin this down in `param_map.json` once.
   **Update:** the ML inputs are now `R, A, CC, VC, T, N`. `T`/`N` select the part file (`N6AShell`, `N6TAShell`, `N0AShell`, `N0TAShell`, ...) and are *not* equations in any part, so SwGen must never see those columns: `src/stp_preflight.py` checks each point against the constraints of its own configuration (`src/constraints.py`) and writes one `R,A,CC,VC` CSV per configuration.
4. **Data confidentiality with Gemini** — everything tool calls return goes to Google. Keep tool outputs to params/status (no raw sim data), and check whether your project/NASA rules allow it. Gemini **free tier may use prompts for training**; use a paid key if in doubt.
5. **Cross-machine handoff** — decide how validated STEPs reach the Abaqus PC (shared drive/OneDrive/scp). The Abaqus script hardcodes `C:\Pablo\...` paths — parametrize it when you get to stage 2.
6. **Batch as the unit of provenance** — each batch records which model version/acquisition produced it, so fine-tuning later knows where points came from.
7. **Human gate** — agent finalizes the batch but a human approves sending to simulation (at least initially).
8. **SolidWorks dialogs/licensing popups** block automation; run a smoke test after SW updates.

---

## Implementation order
0. Save this plan into the repo as `docs/STP_AGENT_PIPELINE_PLAN.md` (sits next to `ML_PLAN.md` / `docs/*_DESIGN.md`).
1. `SwGen probe` — attach to SW, print version. (Proves Part 1 setup.)
2. `SwGen generate` on 1 hard-coded point → STEP matches the old VBA output.
3. Batch mode + JSON results + read-back + restart handling; run on ~10 points from `test_points.csv`, including deliberately infeasible ones.
4. MCP server with `check_constraints` + `batch_status` only; test with **MCP Inspector** (`npx @modelcontextprotocol/inspector python server.py`) — no LLM needed.
5. Add `generate_parts`, `validate_step`, `retry_point`, `finalize_batch`.
6. Gemini agent (CLI first, then `run_agent.py`).

## Verification
- Step 2: open generated STEP in SolidWorks/Abaqus; compare volume to VBA-generated part for the same params.
- Step 3: feed known-bad points → must come back `failed`/`invalid` with a reason, never a STEP marked `ok`.
- Step 4–5: call every tool manually in MCP Inspector and confirm outputs.
- Step 6: agent run on a 10-point batch with 2 injected bad points → `report.json` shows 8 validated, 2 rejected with reasons; import one STEP into the Abaqus script on the other machine to confirm it meshes.
