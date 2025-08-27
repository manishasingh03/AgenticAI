# utils.py
import os
import json
import pandas as pd
import re
from datetime import datetime

DATA_FILE_PATH = 'banking_current_issues.json'

SAMPLE_ISSUES_JSON = """
[
  {
    "issue_id": "ISSUE-0001",
    "issue_type": "transaction_error",
    "creation_date": "2024-07-31",
    "issue_description": "Customer reported a duplicate charge for a purchase made on July 29, 2024. The original transaction was for $50, but two charges of $50 appear on their statement. The second charge was posted on July 30, 2024. The customer claims they only made one purchase.",
    "comments_log": "2024-07-31: We have received the customer's report and are investigating.\\n2024-08-01: An agent contacted the customer to confirm details.",
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
    "comments_log": "2024-07-31: Fraud team received alert.\\n2024-08-01: Customer contacted to verify transaction. Customer stated they were on vacation.\\n2024-08-02: Customer called to report card stolen. Conflicting information.",
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
    "comments_log": "2024-07-30: Customer support received call. Troubleshooting began.\\n2024-07-30: Account was locked due to too many failed login attempts. Customer was able to reset password and log in.",
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

issues_df = pd.DataFrame()

def load_issues_data():
    """
    Load issues JSON (or fallback) into global issues_df; return list of issue_ids.
    """
    global issues_df
    try:
        if not os.path.exists(DATA_FILE_PATH):
            data = json.loads(SAMPLE_ISSUES_JSON)
            print(f"⚠️  '{DATA_FILE_PATH}' not found — using sample data.")
        else:
            with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded data from {DATA_FILE_PATH}.")

        issues_df = pd.DataFrame(data)

        required = ['issue_id', 'creation_date', 'issue_description', 'comments_log',
                    'remediation_plan', 'remediation_completion_date', 'due_date',
                    'status', 'department', 'severity', 'root_cause']
        if not all(col in issues_df.columns for col in required):
            print(f"❌ Missing required columns. Required = {required}")
            return []

        issues_df['creation_date_dt'] = pd.to_datetime(issues_df['creation_date'], errors='coerce')
        issues_df['remediation_completion_date_dt'] = pd.to_datetime(issues_df['remediation_completion_date'], errors='coerce')
        issues_df['due_date_dt'] = pd.to_datetime(issues_df['due_date'], errors='coerce')

        issues_df.sort_values(by='creation_date_dt', inplace=True)

        return issues_df['issue_id'].dropna().unique().tolist()
    except Exception as e:
        print(f"❌ Error loading issues: {e}")
        return []

def get_issue_by_id(issue_id: str) -> dict:
    """
    Return a single issue row as dict (with *_dt fields).
    """
    global issues_df
    if issues_df.empty:
        return {}
    row = issues_df[issues_df['issue_id'] == issue_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()

def generate_issue_timeline_markdown(issue_data, detected_contradictions, contradiction_reasons):
    """
    Render a vertical timeline as Markdown+CSS.
    """
    timeline_events = []

    # 1) Creation
    timeline_events.append({
        'date': issue_data['creation_date_dt'],
        'description': f"**Issue Created** (ID: {issue_data['issue_id']})",
        'type': 'creation'
    })

    # 2) Comments with date parsing
    comments = str(issue_data['comments_log']).split('\n')
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
            if comment_text.lower() in str(reason).lower() or ctype.lower() in comment_text.lower():
                event_type = "contradiction_comment"
                event_desc = f"**{event_desc}** (Potential {ctype})"

        if comment_date:
            timeline_events.append({
                'date': comment_date,
                'description': event_desc,
                'type': event_type
            })

    # 3) Remediation Completed
    if pd.notna(issue_data.get('remediation_completion_date_dt')):
        event_type = "completion"
        if "Remediation Consistency Issue" in detected_contradictions:
            event_type = "contradiction_completion"
        timeline_events.append({
            'date': issue_data['remediation_completion_date_dt'],
            'description': f"**Remediation Completed** (Status: {issue_data['status']})",
            'type': event_type
        })

    # 4) Due Date
    if pd.notna(issue_data.get('due_date_dt')):
        timeline_events.append({
            'date': issue_data['due_date_dt'],
            'description': "**Due Date**",
            'type': 'due_date'
        })

    # Sort by date
    timeline_events = sorted([e for e in timeline_events if e['date'] is not None], key=lambda x: x['date'])

    BASE_SPACING_PX = 30
    PIXELS_PER_DAY = 0.8
    MAX_ADDITIONAL_SPACING_PX = 150

    timeline_md = "### Issue Timeline\n\n<div class='timeline-container'>\n"

    previous_date = None
    for event in timeline_events:
        date_str = event['date'].strftime('%Y-%m-%d')
        margin_top_style = ""
        if previous_date:
            delta_days = (event['date'] - previous_date).days
            additional_spacing = min(MAX_ADDITIONAL_SPACING_PX, delta_days * PIXELS_PER_DAY)
            total_spacing = BASE_SPACING_PX + additional_spacing
            margin_top_style = f"margin-top: {total_spacing}px;"

        item_class = f"timeline-item {event['type']}"
        if "contradiction" in event['type']:
            item_class += " timeline-highlight-contradiction"

        timeline_md += f"<div class='{item_class}' style='{margin_top_style}'>"
        timeline_md += f"<div class='timeline-date'>{date_str}</div>"
        timeline_md += f"<div class='timeline-description'>{event['description']}</div>"
        timeline_md += "</div>\n"

        previous_date = event['date']

    timeline_md += "</div>\n"
    return timeline_md
