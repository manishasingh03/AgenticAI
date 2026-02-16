#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gradio as gr
import pandas as pd
from agentic_framework.agents.utils_agent2 import load_issues_data, generate_issue_timeline_markdown
from agent import GenericAgent

# Load your data
issues_df = pd.DataFrame()
issue_ids = load_issues_data()
if issue_ids:
    issues_df = pd.read_json("banking_current_issues.json")

# Define the async Gradio function
async def analyze_issue_for_contradictions(issue_id):
    selected_issue = issues_df[issues_df['issue_id'] == issue_id].iloc[0].to_dict()
    agent = GenericAgent(model="gemini")
    contradictions, reasons, explanations = await agent.analyze_issue(selected_issue)

    summary = "**Contradictions Found:** " + ("YES" if contradictions else "NO")
    details = f"### Issue ID: {issue_id}\n" + selected_issue["issue_description"]
    contradiction_md = "\n".join([f"### {t}\n{r}\n\n{e}" for t, r, e in zip(contradictions, reasons, explanations)])
    timeline_md = generate_issue_timeline_markdown(selected_issue, contradictions, reasons)

    return summary, details, contradiction_md, timeline_md

# Build Gradio interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Agentic Audit Assistant")

        dropdown = gr.Dropdown(choices=issue_ids, label="Select Issue ID")
        analyze_btn = gr.Button("Analyze")

        summary_out = gr.Markdown()
        details_out = gr.Markdown()
        contradiction_out = gr.Markdown()
        timeline_out = gr.Markdown()

        analyze_btn.click(
            analyze_issue_for_contradictions,
            inputs=[dropdown],
            outputs=[summary_out, details_out, contradiction_out, timeline_out]
        )

    return demo

# Launch
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch()

