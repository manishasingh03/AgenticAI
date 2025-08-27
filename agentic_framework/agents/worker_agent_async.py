# worker_agent_async.py
from agents import Agent
from tools_gemini_async import (
    detect_semantic_contradiction,
    detect_remediation_consistency,
    detect_cross_doc_contradiction,
    explain_contradiction,
)

worker_agent = Agent(
    name="IssueWorkerAgent",
    instructions=(
        "You are a banking QA worker.\n"
        "You will receive an ISSUE_JSON block inside the user prompt.\n"
        "Steps:\n"
        "1) Parse ISSUE_JSON (a single issue object).\n"
        "2) Call the three detection tools to check for contradictions.\n"
        "3) For each detected contradiction, call the explanation tool.\n"
        "4) Return STRICT JSON ONLY with keys:\n"
        '{"contradictions":[], "reasons":[], "explanations":[]} (no extra prose, no markdown).'
    ),
    tools=[
        detect_semantic_contradiction,
        detect_remediation_consistency,
        detect_cross_doc_contradiction,
        explain_contradiction,
    ],
)
