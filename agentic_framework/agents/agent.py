#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# agent.py

from agentic_framework.agents.tool_agent2 import (
    detect_semantic_contradiction,
    detect_remediation_consistency,
    detect_cross_doc_contradiction,
    explain_contradiction
)

class GenericAgent:
    def __init__(self, model="gemini"):
        self.model = model
        self.tools = {
            "semantic": detect_semantic_contradiction,
            "remediation": detect_remediation_consistency,
            "crossdoc": detect_cross_doc_contradiction,
        }

    async def analyze_issue(self, issue):
        contradictions = []
        reasons = []
        explanations = []

        # Run all tools
        semantic, remediation, crossdoc = await asyncio.gather(
            self.tools["semantic"](issue["issue_description"], issue["comments_log"], self.model),
            self.tools["remediation"](issue["issue_description"], issue["comments_log"], issue["root_cause"], issue["remediation_plan"], issue["status"], self.model),
            self.tools["crossdoc"](issue["issue_description"], issue["comments_log"], self.model)
        )

        for flag, ctype, reason in [semantic, remediation, crossdoc]:
            if flag:
                contradictions.append(ctype)
                reasons.append(reason)
                explanation = await explain_contradiction(
                    issue_id=issue["issue_id"],
                    contradiction_type=ctype,
                    issue_description=issue["issue_description"],
                    comments_log=issue["comments_log"],
                    root_cause=issue["root_cause"],
                    remediation_plan=issue["remediation_plan"],
                    status=issue["status"],
                    model=self.model
                )
                explanations.append(explanation)

        return contradictions, reasons, explanations

