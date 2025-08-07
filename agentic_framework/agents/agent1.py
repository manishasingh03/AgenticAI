from agents import Agent,Runner,OpenAIChatCompletionsModel,function_tool
from openai import AsyncOpenAI
from agentic_framework.tools.embeddings_cloudflare import get_embedding
from agentic_framework.tools.gemini_llm import ask_gemini
from agentic_framework.tools.weaviate_kb import fetch_similar_issues, close_client
from agentic_framework.prompts import REACT_INSTRUCTIONS
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import asyncio
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
load_dotenv()


class PlannerInput(BaseModel):
    issue_id: str
    issue_type: str
    issue_description: str
    resolution: str


class PlannerOutput(BaseModel):
    issue_id: str
    issue_type: str
    remediation_plan: str
    status: str


# Read credentials
open_ai_api_key = os.getenv("OPENAI_API_KEY")
open_ai_base_url = os.getenv("OPENAI_API_BASE")
# Create client to use Gemini
gemini_client = AsyncOpenAI(base_url=open_ai_base_url,api_key=open_ai_api_key)
gemini_model = OpenAIChatCompletionsModel(model='gemini-2.5-flash',openai_client=gemini_client)

@function_tool
async def recommend_solutions(issue: PlannerInput) -> str:
    try:
        issue_dict = issue.model_dump()
        issue_id = issue_dict.get("issue_id", "unknown")
        issue_description = issue_dict.get("issue_description", "")
        combined_info = "\n".join(f"{k}: {v}" for k, v in issue_dict.items() if v)

        if not issue_description:
            return f"Issue {issue_id} has no description provided."

        # Summarize with Gemini
        # Note: Span management removed since your Langfuse version might not support it
        prompt = f"""
            Summarize the following banking issue into a single, dense paragraph.
            Focus on the core problem, its root cause, and the actions taken for the final resolution.
            ISSUE DATA:
            {combined_info}
        """
        summary = ask_gemini(prompt)

        # Embed
        query_vector = get_embedding(summary)

        # Retrieve
        similar_issues = fetch_similar_issues(query_vector)

        if not similar_issues:
            return f"No similar resolved issues found for Issue {issue_id}."

        output = [
            f"Issue ID: {i['issue_id']} | Type: {i['issue_type']}\nSuggested Remediation:\n{i['remediation_plan']}"
            for i in similar_issues
        ]
        result = "\n---\n".join(output)
        #print("Result: ",result)
        return result

    except Exception as e:
        raise

worker_agent = Agent(name="IssueWorkerAgent",
    instructions=(
        "Analyze a banking issue dictionary: summarize it, embed it, "
        "and find similar resolved issues to recommend solutions."
    ),tools=[recommend_solutions],
    model=gemini_model)

planner_agent = Agent(
    name="IssuePlannerAgent",
    instructions=REACT_INSTRUCTIONS,
    tools=[
        worker_agent.as_tool(
            tool_name="analyze_issue",
            tool_description="Analyze a banking issue dictionary and recommend remediation suggestions. Give me in a single paragraph."
        )
    ],
    model=gemini_model,
)

async def main():
    input_data = {
        "issue_id":"12345",
        "issue_type":"Login Failure",
        "issue_description":"Customer is unable to log in using mobile banking app after the recent update.",
        "resolution":"Customer was advised to reinstall the app and clear cache."}
        # Convert dict to string prompt
    issue_prompt = "\n".join(f"{k}: {v}" for k, v in input_data.items())
    result = await Runner.run(planner_agent,input=issue_prompt)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())       