#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from agentic_framework.agents.llm_wrapper_agent2 import call_llm

async def detect_semantic_contradiction(issue_description, comments_log, model="gemini"):
    system_prompt = "You detect semantic contradictions in financial issue descriptions and comments. Respond with 'NO_CONTRADICTION' or 'SEMANTIC_CONTRADICTION_FOUND: reason'."
    user_prompt = f"""
    Issue Description: "{issue_description}"
    Comments: "{comments_log}"
    """
    response = await call_llm(system_prompt, user_prompt, model)
    if response.startswith("SEMANTIC_CONTRADICTION_FOUND:"):
        return True, "Semantic Contradiction", response.replace("SEMANTIC_CONTRADICTION_FOUND:", "").strip()
    return False, "", ""

async def detect_remediation_consistency(issue_description, comments_log, root_cause, remediation_plan, status, model="gemini"):
    system_prompt = "You evaluate remediation consistency. Respond with 'NO_CONTRADICTION' or 'REMEDIATION_INCONSISTENCY_FOUND: reason'."
    user_prompt = f"""
    Description: {issue_description}
    Comments: {comments_log}
    Root Cause: {root_cause}
    Remediation: {remediation_plan}
    Status: {status}
    """
    response = await call_llm(system_prompt, user_prompt, model)
    if response.startswith("REMEDIATION_INCONSISTENCY_FOUND:"):
        return True, "Remediation Consistency Issue", response.replace("REMEDIATION_INCONSISTENCY_FOUND:", "").strip()
    return False, "", ""

async def detect_cross_doc_contradiction(issue_description, comments_log, model="gemini"):
    system_prompt = "You compare initial issue description with follow-up comments for contradictions. Respond with 'NO_CONTRADICTION' or 'CROSS_DOCUMENT_CONTRADICTION_FOUND: reason'."
    user_prompt = f"""
    Initial Description: "{issue_description}"
    Updates/Comments: "{comments_log}"
    """
    response = await call_llm(system_prompt, user_prompt, model)
    if response.startswith("CROSS_DOCUMENT_CONTRADICTION_FOUND:"):
        return True, "Cross-Document Contradiction", response.replace("CROSS_DOCUMENT_CONTRADICTION_FOUND:", "").strip()
    return False, "", ""

async def explain_contradiction(issue_id, contradiction_type, issue_description, comments_log, root_cause, remediation_plan, status, model="gemini"):
    system_prompt = "You explain detected contradictions in financial issue reports."
    user_prompt = f"""
    Issue ID: {issue_id}
    Type: {contradiction_type}
    Description: {issue_description}
    Comments: {comments_log}
    Root Cause: {root_cause}
    Remediation: {remediation_plan}
    Status: {status}
    """
    return await call_llm(system_prompt, user_prompt, model)

