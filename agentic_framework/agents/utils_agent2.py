#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils.py

import os
import json
import pandas as pd
import re
from datetime import datetime

DATA_FILE_PATH = 'banking_current_issues.json'

# Sample fallback data
SAMPLE_ISSUES_JSON = """
[{"issue_id": "ISSUE-0001", "issue_type": "transaction_error", "creation_date": "2024-07-31",
"issue_description": "Customer reported a duplicate charge for a purchase made on July 29, 2024...",
"comments_log": "2024-07-31: We have received the customer's report and are investigating.",
"remediation_plan": "Issue a credit for the duplicate charge and notify the customer.",
"remediation_completion_date": "2024-08-02", "due_date": "2024-08-05", "status": "Closed",
"department": "Fraud", "severity": "Medium", "root_cause": "A processing glitch caused the transaction to be submitted twice."}]
"""  # You can replace this with your full fallback JSON if needed

issues_df = pd.DataFrame()

def load_issues_data():
    """
    Loads issue data from file or fallback and prepares global issues_df.
    Returns issue_id list.
    """
    global issues_df
    try:
        if not os.path.exists(DATA_FILE_PATH):
            data = json.loads(SAMPLE_ISSUES_JSON)
            print(f"⚠️  {DATA_FILE_PATH} not found. Using sample data.")
        else:
            with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded issues from {DATA_FILE_PATH}.")

        issues_df = pd.DataFrame(data)

        # Convert dates
        issues_df['creation_date_dt'] = pd.to_datetime(issues_df['creation_date'], errors='coerce')
        issues_df['remediation_completion_date_dt'] = pd.to_datetime(issues_df['remediation_completion_date'], errors='coerce')
        issues_df['due_date_dt'] = pd.to_datetime(issues_df['due_date'], errors='coerce')

        issues_df.sort_values(by='creation_date_dt', inplace=True)

        return issues_df['issue_id'].tolist()

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []

def generate_issue_timeline_markdown(issue_data, detected_contradictions, contradiction_reasons):
    """
    Generates a visual Markdown timeline from issue events.
    """
    timeline_events = []

    # 1. Issue Creation
    timeline_events.append({
        'date': issue_data['creation_date_dt'],
        'description': f"**Issue Created** (ID: {issue_data['issue_id']})",
        'type': 'creation'
    })

    # 2. Comments Log
    comments = issue_data['comments_log'].split('\n')
    for comment_text in comments:
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', comment_text)
        comment_date = None
        if date_match:
            try:
                comment_date = datetime.strptime(date_match.group(0), '%Y-%m-%d')
            except ValueError:
                pass

        event_desc = f"Comment: {comment_text.strip()}"
        event_type = "comment"

        for ctype, reason in zip(detected_contradictions, contradiction_reasons):
            if comment_text.lower() in reason.lower() or ctype.lower() in comment_text.lower():
                event_type = "contradiction_comment"
                event_desc = f"**{event_desc}** (Potential {ctype})"

        if comment_date:
            timeline_events.append({
                'date': comment_date,
                'description': event_desc,
                'type': event_type
            })

    # 3. Remediation Completed
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

    # Sort by date
    timeline_events = sorted([e for e in timeline_events if e['date']], key=lambda x: x['date'])

    # Timeline rendering (Markdown with dynamic spacing)
    BASE_SPACING_PX = 30
    PIXELS_PER_DAY = 0.8
    MAX_SPACING = 150

    timeline_md = "### Issue Timeline\n\n<div class='timeline-container'>\n"

    previous_date = None
    for event in timeline_events:
        date_str = event['date'].strftime('%Y-%m-%d')
        spacing_style = ""

        if previous_date:
            delta_days = (event['date'] - previous_date).days
            extra_spacing = min(MAX_SPACING, delta_days * PIXELS_PER_DAY)
            spacing_style = f"margin-top: {BASE_SPACING_PX + extra_spacing}px;"

        item_class = f"timeline-item {event['type']}"
        if "contradiction" in event['type']:
            item_class += " timeline-highlight-contradiction"

        timeline_md += f"<div class='{item_class}' style='{spacing_style}'>"
        timeline_md += f"<div class='timeline-date'>{date_str}</div>"
        timeline_md += f"<div class='timeline-description'>{event['description']}</div>"
        timeline_md += "</div>\n"

        previous_date = event['date']

    timeline_md += "</div>\n"
    return timeline_md

