import pandas as pd
import numpy as np
import asyncio
import os
import json
import gradio as gr
from datetime import datetime, timedelta
import re # For parsing dates from comments
import aiohttp # For making asynchronous HTTP requests to Gemini API

# --- Google Gemini configuration (GLOBAL AND ONLY HERE) ---

#GEMINI_API_KEY = os.getenv("OPENAI_API_KEY") # Using OPENAI_API_KEY env var for Gemini key
#GEMINI_BASE_URL = os.getenv("OPENAI_BASE_URL") # Using OPENAI_BASE_URL env var for Gemini base 
GEMINI_API_KEY = "AIzaSyCP1l10Tf2xJK15eVROCBgU4r5CxUXjwBs"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
LLM_MODEL = "gemini-2.5-flash-preview-05-20"

# --- Data File Path ---
DATA_FILE_PATH = 'banking_current_issues.json' # Make sure this file exists in the same directory

# --- Global DataFrame to store loaded issues ---

issues_df = pd.DataFrame()

# Sample data for fallback if the data file is not found
SAMPLE_ISSUES_JSON = """
[
  {
    "issue_id": "ISSUE-0001",
    "issue_type": "transaction_error",
    "creation_date": "2024-07-31",
    "issue_description": "Customer reported a duplicate charge for a purchase made on July 29, 2024. The original transaction was for $50, but two charges of $50 appear on their statement. The second charge was posted on July 30, 2024. The customer claims they only made one purchase.",
    "comments_log": "2024-07-31: We have received the customer's report and are investigating. 2024-08-01: An agent contacted the customer to confirm details.",
    "remediation_plan": "Issue a credit for the duplicate charge and notify the customer.",
    "remediation_completion_date": "2024-08-02",
    "due_date": "2024-08-05",
    "status": "Closed",
    "department": "Fraud",
    "severity": "Medium",
    "root_cause": "A processing glitch caused the transaction to be submitted twice."
  },
  {
    "issue_id": "ISSUE-0002",
    "issue_type": "fraud_alert",
    "creation_date": "2024-07-31",
    "issue_description": "A fraud alert was triggered for a high-value international transaction of $1500. The customer claims they were on vacation and made the purchase, but their card was reported stolen a day later. This seems contradictory.",
    "comments_log": "2024-07-31: Fraud team received alert. 2024-08-01: Customer contacted to verify transaction. Customer stated they were on vacation. 2024-08-02: Customer called to report card stolen. Conflicting information.",
    "remediation_plan": "Investigate transaction history and communication logs to resolve the contradictory statements.",
    "remediation_completion_date": null,
    "due_date": "2024-08-07",
    "status": "Open",
    "department": "Fraud",
    "severity": "High",
    "root_cause": null
  },
  {
    "issue_id": "ISSUE-0003",
    "issue_type": "account_access",
    "creation_date": "2024-07-30",
    "issue_description": "Customer unable to log into their online banking account. They state their password is correct but the system returns an 'invalid login' error. They claim they tried multiple times this morning and successfully logged in yesterday evening.",
    "comments_log": "2024-07-30: Customer support received call. Troubleshooting began. 2024-07-30: Account was locked due to too many failed login attempts. Customer was able to reset password and log in.",
    "remediation_plan": "Assisted customer with password reset.",
    "remediation_completion_date": "2024-07-30",
    "due_date": "2024-08-01",
    "status": "Closed",
    "department": "Customer Support",
    "severity": "Low",
    "root_cause": "User entered wrong password multiple times, leading to an automatic account lock."
  }
]
"""

# --- LLM API Call Function ---
async def call_llm_api(system_prompt, user_prompt):
    """
    Makes an asynchronous POST request to the Gemini API.
    Uses exponential backoff for retries on transient errors.
    """
    # Define headers and payload for the Gemini API call
    headers = {
        "Content-Type": "application/json"
    }

    prompt = f"{system_prompt}\n\n{user_prompt}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0, # Keep low for factual extraction to minimize creativity
            "maxOutputTokens": 2048,
        }
    }

    retries = 0
    max_retries = 5
    base_delay = 1.0  # seconds

    async with aiohttp.ClientSession() as session:
        while retries < max_retries:
            try:
                # Add the API key to the URL as a query parameter
                api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

                async with session.post(api_url_with_key, headers=headers, data=json.dumps(payload)) as response:
                    # Raise an exception for HTTP error codes (4xx or 5xx)
                    response.raise_for_status()

                    data = await response.json()

                    # Extract the generated text
                    if 'candidates' in data and data['candidates']:
                        return data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        print("Error: No candidates found in API response.")
                        return "Error: Could not generate a response. No candidates found."

            except aiohttp.ClientResponseError as e:
                # Handle specific HTTP errors with retries
                print(f"HTTP Error {e.status} (Attempt {retries + 1}/{max_retries}): {e.message}. Retrying...")
                retries += 1
                if retries < max_retries:
                    delay = base_delay * (2 ** retries)
                    await asyncio.sleep(delay)
                else:
                    return f"Failed to get a response after {max_retries} attempts: HTTP Error {e.status} - {e.message}"

            except aiohttp.ClientError as e:
                # Handle other aiohttp client errors
                print(f"Aiohttp client error (Attempt {retries + 1}/{max_retries}): {e}. Retrying...")
                retries += 1
                if retries < max_retries:
                    delay = base_delay * (2 ** retries)
                    await asyncio.sleep(delay)
                else:
                    return f"Failed to get a response after {max_retries} attempts: {e}"

            except Exception as e:
                return f"An unexpected error occurred during API call: {e}"

    return "Failed to get a response due to multiple retries."

# --- LLM-Based Contradiction Detection Functions ---

async def detect_semantic_contradiction_llm(issue_description, comments_log):
    """
    Uses LLM to detect semantic contradictions within the issue narrative (description + comments).
    """
    system_prompt = "You are a highly analytical assistant specialized in identifying inconsistencies and contradictions in financial operational reports. Be precise and provide direct answers according to the requested format. If no contradiction, state 'NO_CONTRADICTION'."
    user_prompt = f"""
    Analyze the following banking issue report description and its associated comment log for any semantic contradictions or logical inconsistencies.
    Consider if any statements, facts, or implications within the description or comments conflict with each other.

    Issue Description: "{issue_description}"
    Comment Log: "{comments_log}"

    If a semantic contradiction is found, start your response with "SEMANTIC_CONTRADICTION_FOUND: " followed by a brief explanation.
    Otherwise, respond with "NO_CONTRADICTION".
    """
    response = await call_llm_api(system_prompt, user_prompt)
    if response and response.startswith("SEMANTIC_CONTRADICTION_FOUND:"):
        return True, "Semantic Contradiction", response.replace("SEMANTIC_CONTRADICTION_FOUND: ", "")
    return False, "", ""

async def detect_remediation_consistency_llm(issue_description, comments_log, root_cause, remediation_plan, status):
    """
    Uses LLM to evaluate if the remediation plan seems effective/consistent with the status
    based on the comments, inferring potential inconsistencies.
    """
    system_prompt = "You are a highly analytical assistant specialized in identifying inconsistencies and contradictions in financial operational reports. Be precise and provide direct answers according to the requested format. If no contradiction, state 'NO_CONTRADICTION'."
    user_prompt = f"""
    Evaluate the effectiveness and consistency of the remediation for the following banking issue.
    Consider if the comments log indicates that the remediation plan successfully addressed the root cause, or if there are signs of recurrence, ineffectiveness, or a mismatch with the current status.

    Issue Description: "{issue_description}"
    Root Cause: "{root_cause}"
    Remediation Plan: "{remediation_plan}"
    Current Status: "{status}"
    Comment Log: "{comments_log}"

    If an inconsistency is found (e.g., remediation failed, issue recurred, status misleading), start your response with "REMEDIATION_INCONSISTENCY_FOUND: " followed by a brief explanation.
    Otherwise, respond with "NO_CONTRADICTION".
    """
    response = await call_llm_api(system_prompt, user_prompt)
    if response and response.startswith("REMEDIATION_INCONSISTENCY_FOUND:"):
        return True, "Remediation Consistency Issue", response.replace("REMEDIATION_INCONSISTENCY_FOUND: ", "")
    return False, "", ""

async def detect_cross_document_contradiction_llm(issue_description, comments_log):
    """
    Uses LLM to simulate cross-document contradiction by comparing the initial issue
    description (as one source) with the evolving comment log (as another source).
    """
    system_prompt = "You are a highly analytical assistant specialized in identifying inconsistencies and contradictions in financial operational reports. Be precise and provide direct answers according to the requested format. If no contradiction, state 'NO_CONTRADICTION'."
    user_prompt = f"""
    Compare the initial 'Issue Description' with the subsequent 'Comment Log' for any factual discrepancies or conflicts.
    Treat the Issue Description as an initial report and the Comment Log as ongoing updates.

    Issue Description (Source A): "{issue_description}"
    Comment Log (Source B - updates): "{comments_log}"

    If a factual contradiction between Source A and Source B is found, start your response with "CROSS_DOCUMENT_CONTRADICTION_FOUND: " followed by a brief explanation.
    Otherwise, respond with "NO_CONTRADICTION".
    """
    response = await call_llm_api(system_prompt, user_prompt)
    if response and response.startswith("CROSS_DOCUMENT_CONTRADICTION_FOUND:"):
        return True, "Cross-Document Contradiction", response.replace("CROSS_DOCUMENT_CONTRADICTION_FOUND: ", "")
    return False, "", ""

async def explain_contradiction_llm(issue_id, detected_type, issue_description, comments_log, root_cause, remediation_plan, status):
    """
    Uses LLM to elaborate on a detected contradiction.
    """
    system_prompt = "You are a highly analytical assistant specialized in identifying inconsistencies and contradictions in financial operational reports. Be precise and provide direct answers according to the requested format."
    user_prompt = f"""
    An inconsistency of type '{detected_type}' was identified for Issue ID {issue_id}.
    Provide a detailed explanation of this contradiction, referencing the provided context.

    Issue Description: "{issue_description}"
    Root Cause: "{root_cause}"
    Remediation Plan: "{remediation_plan}"
    Current Status: "{status}"
    Comment Log: "{comments_log}"

    Explain the discrepancy clearly and concisely.
    """
    response = await call_llm_api(system_prompt, user_prompt)
    return response

# --- Data Loading and Processing ---
def load_issues_data():
    """
    Loads the banking issues data from a JSON file into a pandas DataFrame.
    If the file is not found, it falls back to using sample data.
    Returns the list of unique issue IDs for the dropdown.
    """
    global issues_df
    data_source_info = ""
    try:
        # Check if the file exists, if not, use sample data
        if not os.path.exists(DATA_FILE_PATH):
            data = json.loads(SAMPLE_ISSUES_JSON)
            data_source_info = f"Warning: The file '{DATA_FILE_PATH}' was not found. Loaded sample data instead."
        else:
            with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data_source_info = f"Data loaded successfully from {DATA_FILE_PATH}."

        issues_df = pd.DataFrame(data)
        
        # Check for required columns and ensure correct dtypes
        required_cols = ['issue_id', 'creation_date', 'issue_description', 'comments_log', 'remediation_plan', 'remediation_completion_date', 'due_date', 'status', 'department', 'severity', 'root_cause']
        if not all(col in issues_df.columns for col in required_cols):
             print(f"Error: The loaded data is missing one or more required columns: {required_cols}. {data_source_info}")
             return []

        issues_df['creation_date_dt'] = pd.to_datetime(issues_df['creation_date'], errors='coerce')
        issues_df['remediation_completion_date_dt'] = pd.to_datetime(issues_df['remediation_completion_date'], errors='coerce')
        issues_df['due_date_dt'] = pd.to_datetime(issues_df['due_date'], errors='coerce')
        
        issues_df = issues_df.sort_values(by='creation_date_dt')
        
        issue_ids = issues_df['issue_id'].unique().tolist()
        
        if not issue_ids:
            print(f"{data_source_info} No unique issue IDs were found.")
        else:
            print(f"{data_source_info} Total issues: {len(issues_df)}")
        
        return issue_ids

    except json.JSONDecodeError:
        print(f"Error: The file {DATA_FILE_PATH} is not a valid JSON file.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


# --- Visualization Helper Function ---
def generate_issue_timeline_markdown(issue_data, detected_contradictions, contradiction_reasons):
    """
    Generates a Markdown string representing a visual timeline of the issue,
    highlighting key events and detected contradictions.
    Spacing is sensitive to time differences.
    """
    timeline_events = []

    # 1. Issue Creation Date
    timeline_events.append({
        'date': issue_data['creation_date_dt'],
        'description': f"**Issue Created** (ID: {issue_data['issue_id']})",
        'type': 'creation'
    })

    # 2. Comments Log (with best-effort date parsing)
    comments = issue_data['comments_log'].split('\n')
    for comment_text in comments:
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', comment_text)
        comment_date = None
        if date_match:
            try:
                comment_date = datetime.strptime(date_match.group(0), '%Y-%m-%d')
            except ValueError:
                pass # If date parsing fails, treat as undated comment

        event_desc = f"Comment: {comment_text.strip()}"
        event_type = "comment"

        # Check if this comment is related to a detected contradiction
        for ctype, reason in zip(detected_contradictions, contradiction_reasons):
            # Simple check: if comment text is part of the reason, mark it
            if comment_text.lower() in reason.lower() or ctype.lower() in comment_text.lower():
                event_type = "contradiction_comment"
                event_desc = f"**{event_desc}** (Potential {ctype})"
            
        if comment_date: # Only add dated comments to sorted list
            timeline_events.append({
                'date': comment_date,
                'description': event_desc,
                'type': event_type
            })
        else: # Undated comments can be added separately or handled as a block
            pass # Undated comments are currently skipped for precise timeline placement

    # 3. Remediation Completion Date
    if pd.notna(issue_data['remediation_completion_date_dt']):
        event_type = "completion"
        if "Remediation Consistency Issue" in detected_contradictions:
            event_type = "contradiction_completion"
        timeline_events.append({
            'date': issue_data['remediation_completion_date_dt'],
            'description': f"**Remediation Completed** (Status: {issue_data['status']})",
            'type': event_type
        })

    # 4. Due Date
    if pd.notna(issue_data['due_date_dt']):
        timeline_events.append({
            'date': issue_data['due_date_dt'],
            'description': "**Due Date**",
            'type': 'due_date'
        })

    # Sort events by date
    dated_events = sorted([e for e in timeline_events if e['date'] is not None], key=lambda x: x['date'])

    # Constants for dynamic spacing
    BASE_SPACING_PX = 30  # Minimum spacing between events (pixels)
    PIXELS_PER_DAY = 0.8  # How many pixels to add per day difference
    MAX_ADDITIONAL_SPACING_PX = 150 # Cap for additional spacing for very large gaps

    timeline_markdown = "### Issue Timeline\n\n"
    timeline_markdown += "<div class='timeline-container'>\n"

    previous_date = None
    for i, event in enumerate(dated_events):
        date_str = event['date'].strftime('%Y-%m-%d')
        
        margin_top_style = ""
        if previous_date is not None:
            delta_days = (event['date'] - previous_date).days
            # Calculate dynamic spacing
            additional_spacing = min(MAX_ADDITIONAL_SPACING_PX, delta_days * PIXELS_PER_DAY)
            total_spacing = BASE_SPACING_PX + additional_spacing
            margin_top_style = f"margin-top: {total_spacing}px;"
        
        # Apply specific CSS classes based on event type
        item_class = f"timeline-item {event['type']}"
        if "contradiction" in event['type']:
            item_class += " timeline-highlight-contradiction"

        timeline_markdown += f"<div class='{item_class}' style='{margin_top_style}'>"
        timeline_markdown += f"<div class='timeline-date'>{date_str}</div>"
        timeline_markdown += f"<div class='timeline-description'>{event['description']}</div>"
        timeline_markdown += "</div>\n"
        
        previous_date = event['date']
    
    timeline_markdown += "</div>\n"

    return timeline_markdown


# --- Gradio Interface Logic ---

async def analyze_issue_for_contradictions(issue_id):
    """
    Analyzes a selected issue for contradictions using LLM-based methods.
    Returns formatted strings for display in Gradio.
    """
    if issues_df.empty:
        return "Error: Data not loaded. Please check file path and reload the page.", "", "", ""

    selected_issue = issues_df[issues_df['issue_id'] == issue_id].iloc[0]

    # Display basic issue details
    issue_details_markdown = f"""
    ### Issue Details: {selected_issue['issue_id']}
    - **Issue Type:** {selected_issue['issue_type']}
    - **Department:** {selected_issue['department']}
    - **Severity:** {selected_issue['severity']}
    - **Status:** {selected_issue['status']}
    - **Creation Date:** {selected_issue['creation_date']}
    - **Remediation Completion Date:** {selected_issue['remediation_completion_date'] if pd.notna(selected_issue['remediation_completion_date']) else 'N/A'}
    - **Due Date:** {selected_issue['due_date'] if pd.notna(selected_issue['due_date']) else 'N/A'}
    - **Root Cause:** {selected_issue['root_cause']}

    #### Description:
    {selected_issue['issue_description']}

    #### Comments Log:
    {selected_issue['comments_log']}

    ---
    """

    detected_contradictions = []
    contradiction_reasons = []
    llm_explanations = []

    # Run LLM-based detection tasks concurrently
    tasks = [
        detect_semantic_contradiction_llm(selected_issue['issue_description'], selected_issue['comments_log']),
        detect_remediation_consistency_llm(selected_issue['issue_description'], selected_issue['comments_log'], selected_issue['root_cause'], selected_issue['remediation_plan'], selected_issue['status']),
        detect_cross_document_contradiction_llm(selected_issue['issue_description'], selected_issue['comments_log'])
    ]
    
    results = await asyncio.gather(*tasks)

    # Process results and collect findings
    if results[0][0]: # Semantic Contradiction
        detected_contradictions.append(results[0][1])
        contradiction_reasons.append(results[0][2])
    if results[1][0]: # Remediation Consistency Issue
        detected_contradictions.append(results[1][1])
        contradiction_reasons.append(results[1][2])
    if results[2][0]: # Cross-Document Contradiction
        detected_contradictions.append(results[2][1])
        contradiction_reasons.append(results[2][2])

    # --- Quick Summary Output ---
    summary_text = ""
    if detected_contradictions:
        summary_text = f"**Contradictions Found:** YES ({len(detected_contradictions)} types detected)"
        summary_text += "\n**Types:** " + ", ".join(detected_contradictions)
    else:
        summary_text = "**Contradictions Found:** NO"

    # --- Detailed Contradiction Output ---
    contradiction_output_markdown = ""
    if detected_contradictions:
        contradiction_output_markdown += "### Detailed Contradiction Analysis:\n"
        for i, (ctype, reason) in enumerate(zip(detected_contradictions, contradiction_reasons)):
            contradiction_output_markdown += f"**{i+1}. Type:** {ctype}\n"
            contradiction_output_markdown += f"**Reason for Detection:** {reason}\n"
            
            # Get detailed LLM explanation for each detected contradiction
            explanation = await explain_contradiction_llm(
                selected_issue['issue_id'],
                ctype, # Pass the specific type detected
                selected_issue['issue_description'],
                selected_issue['comments_log'],
                selected_issue['root_cause'],
                selected_issue['remediation_plan'],
                selected_issue['status']
            )
            llm_explanations.append(f"**Detailed LLM Explanation for {ctype}:**\n{explanation}\n")
            contradiction_output_markdown += f"**LLM Explanation:** {explanation}\n\n" # Add to main output

    else:
        contradiction_output_markdown = "### No Detailed Contradictions Found."

    # --- Generate Timeline ---
    timeline_markdown = generate_issue_timeline_markdown(selected_issue, detected_contradictions, contradiction_reasons)


    return summary_text, issue_details_markdown, contradiction_output_markdown, timeline_markdown

# --- Gradio App Definition ---
def create_agent2_interface():
    issue_ids = load_issues_data() # Load data and get IDs when app starts

    with gr.Blocks(title="Agent 2: Quality Assurance & Audit Contradiction Detection",
                    css="""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #f0f2f5; }
        .gradio-container { max-width: 1200px; margin: auto; padding: 20px; background-color: #ffffff; border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
        .gr-box { border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); padding: 20px; background-color: #ffffff; }
        .gr-button { border-radius: 8px; background-color: #007bff; color: white; border: none; padding: 10px 20px; font-weight: 600; transition: background-color 0.3s ease; }
        .gr-button:hover { background-color: #0056b3; }
        h1, h2, h3, h4, h5, h6 { color: #2c3e50; font-weight: 700; }
        .gr-tab-item { border-radius: 8px 8px 0 0; }
        .gr-tab-item.selected { background-color: #e0f2f7; border-bottom: 3px solid #007bff; }
        .gr-markdown { padding: 15px; border-radius: 8px; background-color: #fdfdfd; border: 1px solid #eee; margin-bottom: 10px; }
        .gr-dropdown { border-radius: 8px; }
        .gr-interface-panel { background-color: #ffffff; border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
        #loading_message { text-align: center; color: #007bff; font-weight: 600; margin-top: 15px; }
        #summary_output { background-color: #e6f7ff; border-left: 5px solid #007bff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }

        /* Timeline specific CSS */
        .timeline-container {
            position: relative;
            padding: 20px 0;
            margin: 20px 0;
            border-left: 2px solid #ccc;
            margin-left: 120px; /* Increased space for dates */
        }
        .timeline-item {
            position: relative;
            /* margin-bottom will be set dynamically via style attribute */
            padding-left: 30px;
        }
        .timeline-item::before {
            content: '';
            position: absolute;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            left: -6px;
            top: 5px;
            background-color: #007bff;
            border: 2px solid #fff;
            z-index: 1;
        }
        .timeline-date {
            position: absolute;
            left: -110px; /* Adjusted position to the left of the line */
            width: 100px; /* Increased width for date */
            text-align: right;
            font-weight: bold;
            color: #555;
            top: 0;
        }
        .timeline-description {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #d0e0f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .timeline-highlight-contradiction {
            background-color: #ffe0e0 !important; /* Light red for contradictions */
            border-color: #ff8080 !important;
        }
        .timeline-highlight-contradiction::before {
            background-color: #ff0000 !important; /* Red dot for contradictions */
            border-color: #fff !important;
        }
        .timeline-item.due_date::before {
            background-color: #ffc107 !important; /* Yellow for due date */
            border-color: #fff !important;
        }
        .timeline-item.creation::before {
            background-color: #28a745 !important; /* Green for creation */
            border-color: #fff !important;
        }
        .timeline-item.completion::before {
            background-color: #6f42c1 !important; /* Purple for completion */
            border-color: #fff !important;
        }
        .timeline-item.comment::before {
            background-color: #17a2b8 !important; /* Teal for regular comments */
            border-color: #fff !important;
        }
    """) as demo:
        gr.Markdown("# Agent 2: Quality Assurance & Audit Contradiction Detection")
        gr.Markdown("Select an issue ID from the dropdown to analyze it for contradictions and inconsistencies. This agent assists QA and Audit teams by providing detailed reasoning.")

        with gr.Row(elem_id="input_row"):
            issue_dropdown = gr.Dropdown(
                choices=issue_ids,
                label="Select Issue ID",
                interactive=True,
                value=issue_ids[0] if issue_ids else None # Pre-select first issue if available
            )
            analyze_button = gr.Button("Analyze Issue", variant="primary")

        # Loading indicator message
        loading_message = gr.Markdown("Click 'Analyze Issue' or select an ID to start analysis...", elem_id="loading_message")

        with gr.Column(elem_id="results_column"):
            summary_output = gr.Markdown("## Analysis Summary", elem_id="summary_output")
            
            with gr.Tabs():
                with gr.TabItem("Issue Details"):
                    issue_details_output = gr.Markdown(label="Issue Details")
                with gr.TabItem("Contradiction Analysis"):
                    contradiction_output = gr.Markdown(label="Contradiction Analysis Results")
                with gr.TabItem("Issue Timeline"): # New tab for timeline
                    timeline_output = gr.Markdown(label="Issue Timeline Visualization")

        # Define the event handler for button click
        analyze_button.click(
            fn=lambda: (gr.update(value="Analyzing... Please wait, this may take a moment due to LLM calls."), "", "", "", ""), # Added an empty string for timeline_output
            outputs=[loading_message, summary_output, issue_details_output, contradiction_output, timeline_output], # Added timeline_output
            queue=False # Don't queue this immediate update
        ).then(
            fn=analyze_issue_for_contradictions,
            inputs=issue_dropdown,
            outputs=[summary_output, issue_details_output, contradiction_output, timeline_output] # Added timeline_output
        )
        
        # Define the event handler for dropdown change (similar to button click)
        issue_dropdown.change(
            fn=lambda: (gr.update(value="Analyzing... Please wait, this may take a moment due to LLM calls."), "", "", "", ""), # Added an empty string for timeline_output
            outputs=[loading_message, summary_output, issue_details_output, contradiction_output, timeline_output], # Added timeline_output
            queue=False
        ).then(
            fn=analyze_issue_for_contradictions,
            inputs=issue_dropdown,
            outputs=[summary_output, issue_details_output, contradiction_output, timeline_output] # Added timeline_output
        )

        # Initial analysis when the page loads if an issue is pre-selected
        if issue_ids:
            demo.load(
                fn=lambda: (gr.update(value="Analyzing... Please wait, this may take a moment due to LLM calls."), "", "", "", ""), # Added an empty string for timeline_output
                outputs=[loading_message, summary_output, issue_details_output, contradiction_output, timeline_output], # Added timeline_output
                queue=False
            ).then(
                fn=analyze_issue_for_contradictions,
                inputs=issue_dropdown,
                outputs=[summary_output, issue_details_output, contradiction_output, timeline_output] # Added timeline_output
            )


    return demo

# --- Launch the Gradio App ---
if __name__ == "__main__":
    # Initial data load before launching the app
    _ = load_issues_data()  
    
    if issues_df.empty:
        print("Cannot launch Gradio app: No data loaded. Please check DATA_FILE_PATH and ensure the JSON file is valid.")
    else:
        demo = create_agent2_interface()
        demo.launch(share=True)

