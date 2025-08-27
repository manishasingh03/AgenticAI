# tools_gemini_async.py
from agents import function_tool
from typing import Optional, Tuple
from gemini_http_async import call_gemini_async

@function_tool
async def detect_semantic_contradiction(
    issue_description: str,
    comments_log: str,
) -> Tuple[bool, str, str]:
    """
    Returns (found_flag, type_name, reason)
    """
    system = "You detect semantic contradictions in banking issue reports."
    user = f"""
Analyze for semantic contradictions or logical inconsistencies.

Issue Description: "{issue_description}"
Comment Log: "{comments_log}"

Respond ONLY with:
- "NO_CONTRADICTION"
- or "SEMANTIC_CONTRADICTION_FOUND: <brief reason>"
"""
    out = await call_gemini_async(system, user, temperature=0.0)
    if out.startswith("SEMANTIC_CONTRADICTION_FOUND:"):
        return True, "Semantic Contradiction", out.split(":", 1)[1].strip()
    return False, "", ""

@function_tool
async def detect_remediation_consistency(
    issue_description: str,
    comments_log: str,
    root_cause: Optional[str],
    remediation_plan: Optional[str],
    status: str,
) -> Tuple[bool, str, str]:
    """
    Returns (found_flag, type_name, reason)
    """
    system = "You evaluate if remediation plan and status align with the evidence."
    user = f"""
Evaluate remediation consistency.

Issue Description: "{issue_description}"
Root Cause: "{root_cause}"
Remediation Plan: "{remediation_plan}"
Current Status: "{status}"
Comment Log: "{comments_log}"

Respond ONLY with:
- "NO_CONTRADICTION"
- or "REMEDIATION_INCONSISTENCY_FOUND: <brief reason>"
"""
    out = await call_gemini_async(system, user, temperature=0.0)
    if out.startswith("REMEDIATION_INCONSISTENCY_FOUND:"):
        return True, "Remediation Consistency Issue", out.split(":", 1)[1].strip()
    return False, "", ""

@function_tool
async def detect_cross_doc_contradiction(
    issue_description: str,
    comments_log: str,
) -> Tuple[bool, str, str]:
    """
    Returns (found_flag, type_name, reason)
    """
    system = "You compare initial description (A) vs updates/comments (B) for contradictions."
    user = f"""
Source A (Initial): "{issue_description}"
Source B (Updates/Comments): "{comments_log}"

Respond ONLY with:
- "NO_CONTRADICTION"
- or "CROSS_DOCUMENT_CONTRADICTION_FOUND: <brief reason>"
"""
    out = await call_gemini_async(system, user, temperature=0.0)
    if out.startswith("CROSS_DOCUMENT_CONTRADICTION_FOUND:"):
        return True, "Cross-Document Contradiction", out.split(":", 1)[1].strip()
    return False, "", ""

@function_tool
async def explain_contradiction(
    issue_id: str,
    contradiction_type: str,
    issue_description: str,
    comments_log: str,
    root_cause: Optional[str],
    remediation_plan: Optional[str],
    status: str,
) -> str:
    """
    Returns a concise explanation (text)
    """
    system = "You explain detected contradictions clearly and concisely for QA/audit readers."
    user = f"""
Issue ID: {issue_id}
Type: {contradiction_type}

Issue Description: "{issue_description}"
Root Cause: "{root_cause}"
Remediation Plan: "{remediation_plan}"
Current Status: "{status}"
Comment Log: "{comments_log}"

Write a short, precise explanation (3–6 sentences).
"""
    return await call_gemini_async(system, user, temperature=0.0)