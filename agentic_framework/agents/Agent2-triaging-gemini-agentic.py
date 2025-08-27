import pandas as pd
import numpy as np
import asyncio
import aiohttp
import os
import json
import gradio as gr
from datetime import datetime, timedelta
import re
import contextlib
import signal
import sys
import uuid

# The following imports are from the agentic framework. These are not standard libraries
# and have been removed to make the script runnable without them.
# from agents import Agent, function_tool, OpenAIChatCompletionsModel
# from agentic_framework.prompts import REACT_INSTRUCTIONS
from pydantic import BaseModel, Field

# The openai client is not used as the code directly makes an aiohttp call.
# from openai import AsyncOpenAI


# --- Data and API Configuration ---
# Replace with the path to your data file
DATA_FILE_PATH = 'banking_current_issues.json'

# Load environment variables for the API key
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = "AIzaSyCP1l10Tf2xJK15eVROCBgU4r5CxUXjwBs"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

if not GEMINI_API_KEY:
    raise ValueError(
        "Error: The API key (OPENAI_API_KEY) environment variable is not set.\n\n"
        "Please provide it in one of the following ways:\n"
        "1. Set it as a system environment variable.\n"
        "2. Create a file named .env in the same directory as this script and "
        "add OPENAI_API_KEY='your_key_here' to it.\n"
    )

# --- Graceful Shutdown Handlers ---
async def _cleanup_clients() -> None:
    pass

def _handle_sigint(signum: int, frame: object) -> None:
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_sigint)


# --- Agent Pydantic Models ---
class AuditInput(BaseModel):
    """
    Input model for the Quality Assurance Agent.
    It takes all the necessary data fields to perform a comprehensive audit.
    """
    issue_id: str
    issue_type: str
    issue_description: str
    comments_log: str
    remediation_plan: str
    status: str
    root_cause: str


class ContradictionDetail(BaseModel):
    """
    Model to hold the details of a single detected contradiction.
    """
    type: str
    reason: str
    explanation: str


# --- LLM API Call and Contradiction Detection Functions ---
async def call_llm_api(system_prompt, user_prompt):
    """
    Generic function to call the Gemini API with exponential backoff for retries.
    """
    headers = {"Content-Type": "application/json"}
    prompt = f"{system_prompt}\n\n{user_prompt}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048,}
    }
    retries = 0
    max_retries = 5
    base_delay = 1.0
    async with aiohttp.ClientSession() as session:
        while retries < max_retries:
            try:
                api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
                async with session.post(api_url_with_key, headers=headers, data=json.dumps(payload)) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if 'candidates' in data and data['candidates']:
                        return data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        print("Error: No candidates found in API response.")
                        return "Error: Could not generate a response. No candidates found."
            except aiohttp.ClientResponseError as e:
                print(f"HTTP Error {e.status} (Attempt {retries + 1}/{max_retries}): {e.message}. Retrying...")
                retries += 1
                if retries < max_retries:
                    delay = base_delay * (2 ** retries)
                    await asyncio.sleep(delay)
                else:
                    return f"Failed to get a response after {max_retries} attempts: HTTP Error {e.status} - {e.message}"
            except Exception as e:
                return f"An unexpected error occurred during API call: {e}"
    return "Failed to get a response due to multiple retries."


async def detect_semantic_contradiction_llm(issue_description, comments_log):
    """
    Detects semantic contradictions between an issue description and its comments.
    """
    system_prompt = "You are a highly analytical assistant specialized in identifying inconsistencies and contradictions in financial operational reports. Be precise and provide direct answers according to the requested format. If no contradiction, state 'NO_CONTRADICTION'."
    user_prompt = f"Analyze the following banking issue report description and its associated comment log for any semantic contradictions or logical inconsistencies.\n\nIssue Description: \"{issue_description}\"\nComment Log: \"{comments_log}\"\n\nIf a semantic contradiction is found, start your response with \"SEMANTIC_CONTRADICTION_FOUND: \" followed by a brief explanation. Otherwise, respond with \"NO_CONTRADICTION\"."
    response = await call_llm_api(system_prompt, user_prompt)
    if response and response.startswith("SEMANTIC_CONTRADICTION_FOUND:"):
        return True, "Semantic Contradiction", response.replace("SEMANTIC_CONTRADICTION_FOUND: ", "")
    return False, "", ""

async def detect_remediation_consistency_llm(issue_description, comments_log, root_cause, remediation_plan, status):
    """
    Detects inconsistencies in the remediation plan and status.
    """
    system_prompt = "You are a highly analytical assistant specialized in identifying inconsistencies and contradictions in financial operational reports. Be precise and provide direct answers according to the requested format. If no contradiction, state 'NO_CONTRADICTION'."
    user_prompt = f"Evaluate the effectiveness and consistency of the remediation for the following banking issue.\n\nIssue Description: \"{issue_description}\"\nRoot Cause: \"{root_cause}\"\nRemediation Plan: \"{remediation_plan}\"\nCurrent Status: \"{status}\"\nComment Log: \"{comments_log}\"\n\nIf an inconsistency is found (e.g., remediation failed, issue recurred, status misleading), start your response with \"REMEDIATION_INCONSISTENCY_FOUND: \" followed by a brief explanation. Otherwise, respond with \"NO_CONTRADICTION\"."
    response = await call_llm_api(system_prompt, user_prompt)
    if response and response.startswith("REMEDIATION_INCONSISTENCY_FOUND:"):
        return True, "Remediation Consistency Issue", response.replace("REMEDIATION_INCONSISTENCY_FOUND: ", "")
    return False, "", ""

async def detect_cross_document_contradiction_llm(issue_description, comments_log):
    """
    Detects factual discrepancies between the initial issue description and the comments.
    """
    system_prompt = "You are a highly analytical assistant specialized in identifying inconsistencies and contradictions in financial operational reports. Be precise and provide direct answers according to the requested format. If no contradiction, state 'NO_CONTRADICTION'."
    user_prompt = f"Compare the initial 'Issue Description' with the subsequent 'Comment Log' for any factual discrepancies or conflicts.\n\nIssue Description (Source A): \"{issue_description}\"\nComment Log (Source B - updates): \"{comments_log}\"\n\nIf a factual contradiction between Source A and Source B is found, start your response with \"CROSS_DOCUMENT_CONTRADICTION_FOUND: \" followed by a brief explanation. Otherwise, respond with \"NO_CONTRADICTION\"."
    response = await call_llm_api(system_prompt, user_prompt)
    if response and response.startswith("CROSS_DOCUMENT_CONTRADICTION_FOUND:"):
        return True, "Cross-Document Contradiction", response.replace("CROSS_DOCUMENT_CONTRADICTION_FOUND: ", "")
    return False, "", ""

async def explain_contradiction_llm(issue_id, detected_type, issue_description, comments_log, root_cause, remediation_plan, status):
    """
    Generates a detailed explanation for a detected contradiction.
    """
    system_prompt = "You are a highly analytical assistant specialized in identifying inconsistencies and contradictions in financial operational reports. Be precise and provide direct answers according to the requested format."
    user_prompt = f"An inconsistency of type '{detected_type}' was identified for Issue ID {issue_id}.\nProvide a detailed explanation of this contradiction, referencing the provided context.\n\nIssue Description: \"{issue_description}\"\nRoot Cause: \"{root_cause}\"\nRemediation Plan: \"{remediation_plan}\"\nCurrent Status: \"{status}\"\nComment Log: \"{comments_log}\"\n\nExplain the discrepancy clearly and concisely."
    response = await call_llm_api(system_prompt, user_prompt)
    return response


async def perform_audit(audit_data: AuditInput) -> list[ContradictionDetail]:
    """
    This is the core function for the QA process. It orchestrates the contradiction detection
    and returns a list of contradiction details.
    """
    detected_contradictions = []
    
    # Run LLM-based detection tasks concurrently
    tasks = [
        detect_semantic_contradiction_llm(audit_data.issue_description, audit_data.comments_log),
        detect_remediation_consistency_llm(audit_data.issue_description, audit_data.comments_log, audit_data.root_cause, audit_data.remediation_plan, audit_data.status),
        detect_cross_document_contradiction_llm(audit_data.issue_description, audit_data.comments_log)
    ]
    results = await asyncio.gather(*tasks)

    # Process results and collect findings
    for result in results:
        is_found, ctype, reason = result
        if is_found:
            # Get detailed LLM explanation for each detected contradiction
            explanation = await explain_contradiction_llm(
                audit_data.issue_id,
                ctype,
                audit_data.issue_description,
                audit_data.comments_log,
                audit_data.root_cause,
                audit_data.remediation_plan,
                audit_data.status
            )
            detected_contradictions.append(
                ContradictionDetail(type=ctype, reason=reason, explanation=explanation)
            )
        
    return detected_contradictions


# --- Gradio UI Functions ---
def load_data_to_df(data_file_path):
    """Load data from a JSON file into a pandas DataFrame."""
    try:
        with open(data_file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        return pd.DataFrame()

df_banking_issues = load_data_to_df(DATA_FILE_PATH)

def process_issue_data_for_display(issue_id):
    """
    Retrieves and formats a specific issue's data for display in the Gradio UI.
    """
    if df_banking_issues.empty:
        return "Data not loaded.", "", "", "", "", ""
        
    issue_data = df_banking_issues[df_banking_issues['issue_id'] == issue_id].iloc[0]

    issue_info = (
        f"**Issue ID:** {issue_data['issue_id']}\n"
        f"**Issue Type:** {issue_data['issue_type']}\n"
        f"**Status:** {issue_data['status']}\n"
    )
    
    issue_details = f"**Description:** {issue_data['issue_description']}"
    comments_log = f"**Comment Log:** {issue_data['comments_log']}"
    remediation_plan = f"**Remediation Plan:** {issue_data['remediation_plan']}"
    root_cause = f"**Root Cause:** {issue_data['root_cause']}"
    
    return issue_info, issue_details, comments_log, remediation_plan, root_cause

async def audit_issue_button_click(issue_id):
    """
    Button click handler that performs an audit.
    """
    if df_banking_issues.empty:
        return "No data loaded. Cannot perform audit."

    issue_data = df_banking_issues[df_banking_issues['issue_id'] == issue_id].iloc[0]
    
    # Create the AuditInput model from the DataFrame row
    audit_input = AuditInput(
        issue_id=issue_data['issue_id'],
        issue_type=issue_data['issue_type'],
        issue_description=issue_data['issue_description'],
        comments_log=issue_data['comments_log'],
        remediation_plan=issue_data['remediation_plan'],
        status=issue_data['status'],
        root_cause=issue_data['root_cause']
    )
    
    try:
        # Call the core audit function directly
        contradictions = await perform_audit(audit_input)
        
        # Format the output for Gradio display
        output_text = f"## Audit Report for Issue ID: {audit_input.issue_id}\n\n"
        
        if contradictions:
            output_text += "**Audit Status:** Contradictions Found\n\n"
            output_text += "### Detected Contradictions:\n"
            for i, contra in enumerate(contradictions):
                output_text += f"**{i+1}. {contra.type}**\n"
                output_text += f"**Reason:** {contra.reason}\n"
                output_text += f"**Explanation:** {contra.explanation}\n\n"
        else:
            output_text += "**Audit Status:** No Contradictions Found\n"
        
        return output_text

    except Exception as e:
        return f"An error occurred during the audit: {e}"


# --- Gradio UI Layout ---
with gr.Blocks(title="Agentic Banking Issue QA") as demo:
    gr.Markdown("# Banking Issue Auditing System")
    gr.Markdown(
        "This application uses an AI agent to perform a quality assurance audit on banking issues. "
        "Select an issue ID and click 'Audit Issue' to check for contradictions."
    )
    
    # Get a list of all issue IDs for the dropdown
    issue_ids = df_banking_issues['issue_id'].tolist() if not df_banking_issues.empty else []
    
    with gr.Row():
        with gr.Column(scale=1):
            issue_selector = gr.Dropdown(
                choices=issue_ids,
                label="Select Issue ID",
                interactive=True
            )
            audit_button = gr.Button("Audit Issue", interactive=True)
            
        with gr.Column(scale=2):
            issue_info = gr.Markdown("### Issue Details")
            issue_description_field = gr.Markdown("")
            comments_log_field = gr.Markdown("")
            remediation_plan_field = gr.Markdown("")
            root_cause_field = gr.Markdown("")

    audit_result = gr.Markdown("### Audit Result")
    
    # Event Handlers
    issue_selector.change(
        fn=process_issue_data_for_display,
        inputs=issue_selector,
        outputs=[issue_info, issue_description_field, comments_log_field, remediation_plan_field, root_cause_field]
    )
    
    audit_button.click(
        fn=audit_issue_button_click,
        inputs=issue_selector,
        outputs=audit_result
    )

if __name__ == "__main__":
    demo.launch()