# app.py
import json
import gradio as gr
from datetime import datetime
import pandas as pd  # for type checks
from agents import Runner
from worker_agent_async import worker_agent
from utils_async import load_issues_data, get_issue_by_id, generate_issue_timeline_markdown
import os

os.environ["OPENAI_API_KEY"] = ["GEMINI_API_KEY"]

APP_TITLE = "Agentic QA — OpenAI Agents (Planner, async) + Gemini (Analysis via async HTTP Tools)"

TIMELINE_CSS = """
/* Timeline CSS */
.timeline-container {
    position: relative;
    padding: 20px 0;
    margin: 20px 0;
    border-left: 2px solid #ccc;
    margin-left: 120px;
}
.timeline-item { position: relative; padding-left: 30px; }
.timeline-item::before {
    content: ''; position: absolute; width: 10px; height: 10px; border-radius: 50%;
    left: -6px; top: 5px; background-color: #007bff; border: 2px solid #fff; z-index: 1;
}
.timeline-date {
    position: absolute; left: -110px; width: 100px; text-align: right; font-weight: bold; color: #555; top: 0;
}
.timeline-description {
    background-color: #f0f8ff; padding: 10px; border-radius: 8px; border: 1px solid #d0e0f0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.timeline-highlight-contradiction { background-color: #ffe0e0 !important; border-color: #ff8080 !important; }
.timeline-item.due_date::before { background-color: #ffc107 !important; }
.timeline-item.creation::before { background-color: #28a745 !important; }
.timeline-item.completion::before { background-color: #6f42c1 !important; }
.timeline-item.comment::before { background-color: #17a2b8 !important; }
"""

def parse_agent_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {"raw_output": text}

def sanitize_issue_for_agent(issue: dict) -> dict:
    """
    Create a JSON-safe copy:
    - drop any *_dt keys
    - convert pandas.Timestamp / datetime to YYYY-MM-DD strings
    """
    safe = {}
    for k, v in issue.items():
        if k.endswith("_dt"):
            continue  # agent doesn't need internal datetime columns
        if isinstance(v, (pd.Timestamp, datetime)):
            # Convert to simple date string (or ISO if you prefer)
            safe[k] = None if pd.isna(v) else v.strftime("%Y-%m-%d")
        else:
            safe[k] = v
    return safe

async def run_agent_for_issue(issue_id: str):
    if not issue_id:
        return "Select an issue.", "", "", ""

    issue = get_issue_by_id(issue_id)
    if not issue:
        return "Issue not found.", "", "", ""

    # Build a JSON-safe payload ONLY for the agent
    issue_for_agent = sanitize_issue_for_agent(issue)
    issue_json = json.dumps(issue_for_agent, ensure_ascii=False, indent=2)

    prompt = (
        "Analyze the provided issue payload for contradictions. "
        "Call tools as needed and return STRICT JSON with keys: "
        '{"contradictions":[], "reasons":[], "explanations":[]}. No prose.\n\n'
        "ISSUE_JSON:\n```json\n" + issue_json + "\n```"
    )

    # Async agent run (Runner.run does not accept data=..., so we inline JSON above)
    result = await Runner.run(worker_agent, prompt)
    final_text = result.final_output or ""
    parsed = parse_agent_json(final_text)

    # Build the panels (use original 'issue' with *_dt fields so the timeline spacing works)
    details_md = f"""
### Issue Details: {issue.get('issue_id')}
- **Issue Type:** {issue.get('issue_type')}
- **Department:** {issue.get('department')}
- **Severity:** {issue.get('severity')}
- **Status:** {issue.get('status')}
- **Creation Date:** {issue.get('creation_date')}
- **Remediation Completion Date:** {issue.get('remediation_completion_date')}
- **Due Date:** {issue.get('due_date')}
- **Root Cause:** {issue.get('root_cause')}

#### Description:
{issue.get('issue_description')}

#### Comments Log:
{issue.get('comments_log')}
"""

    contradictions = []
    reasons = []
    if isinstance(parsed, dict):
        contradictions = parsed.get("contradictions", [])
        reasons = parsed.get("reasons", [])

    timeline_md = generate_issue_timeline_markdown(issue, contradictions, reasons)

    # Contradictions & output panels
    if isinstance(parsed, dict) and "raw_output" not in parsed:
        contradictions_md = "### Detected Contradictions\n"
        if contradictions:
            for i, (t, r) in enumerate(zip(contradictions, reasons), start=1):
                contradictions_md += f"**{i}. Type:** {t}\n\n**Reason:** {r}\n\n"
        else:
            contradictions_md += "_None detected._"
        output_md = f"### Agent JSON Output\n```json\n{json.dumps(parsed, indent=2)}\n```"
    else:
        contradictions_md = "### Detected Contradictions\n_Raw agent output could not be parsed as JSON._"
        output_md = f"### Agent Raw Output\n\n{final_text}"

    return (
        f"**Analysis complete for {issue_id}.**",
        details_md,
        contradictions_md,
        timeline_md + "\n\n" + output_md
    )

def build_ui():
    issue_ids = load_issues_data()

    with gr.Blocks(title=APP_TITLE, css=TIMELINE_CSS) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown("Select an issue from the JSON file, run the agentic analysis (OpenAI planner + Gemini tools), and view the timeline.")

        with gr.Row():
            issue_dropdown = gr.Dropdown(
                choices=issue_ids, 
                label="Select Issue ID", 
                value=issue_ids[0] if issue_ids else None
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Row():
            status_txt = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Issue Details"):
                details_out = gr.Markdown()
            with gr.Tab("Contradiction Analysis"):
                contradictions_out = gr.Markdown()
            with gr.Tab("Timeline + Agent Output"):
                timeline_out = gr.Markdown()

        async def _start_msg():
            return "Analyzing... please wait.", "", "", ""

        analyze_btn.click(
            _start_msg,
            inputs=[],
            outputs=[status_txt, details_out, contradictions_out, timeline_out],
            queue=False
        ).then(
            run_agent_for_issue,  # async
            inputs=[issue_dropdown],
            outputs=[status_txt, details_out, contradictions_out, timeline_out]
        )

        # Auto-run for first item on load (if available)
        if issue_ids:
            demo.load(
                _start_msg,
                inputs=[],
                outputs=[status_txt, details_out, contradictions_out, timeline_out],
                queue=False
            ).then(
                run_agent_for_issue,  # async
                inputs=[issue_dropdown],
                outputs=[status_txt, details_out, contradictions_out, timeline_out]
            )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    # For local dev, a public link is handy; on HF Spaces, host/port may be managed by the platform.
    ui.launch(server_name="0.0.0.0", server_port=7860, share=True)

