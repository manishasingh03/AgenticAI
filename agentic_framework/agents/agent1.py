from agents import Agent,Runner,OpenAIChatCompletionsModel,function_tool
from openai import AsyncOpenAI
from agentic_framework.tools.embeddings_cloudflare import get_embedding
from agentic_framework.tools.gemini_llm import ask_gemini
from agentic_framework.tools.weaviate_kb import fetch_similar_issues, close_client
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import asyncio
import sys
from typing import Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
load_dotenv()

#main variables
RECOMMEND_SOLUTIONS_PROMPT = "recommend_solutions_prompt.j2"
REACT_INSTRUCTIONS="react_instructions.j2"
JUDGE_ANSWER_PROMPT="judge_answer_prompt.j2"
WORKER_AGENT_INSTRUCTIONS="worker_agent_instructions.j2"

# Set up the environment and loader at top level (ideally singleton/shared)
_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
env = Environment(
    loader=FileSystemLoader(_PROMPT_DIR),
    autoescape=select_autoescape(['j2'])
)

def load_prompt(name, **kwargs):
    """
    Render a Jinja2 prompt template.
    :param name: template file name, e.g., 'planner_agent_instructions.j2'
    :param kwargs: template context variables
    """
    template = env.get_template(name)
    return template.render(**kwargs)

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

class JudgeInput(BaseModel):
    issue: dict  # Or PlannerInput type for stricter typing
    other_input: Optional[dict]
    task_description: str
    model_output: str

class JudgeOutput(BaseModel):
    is_correct: bool
    feedback: str

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
        prompt = load_prompt(RECOMMEND_SOLUTIONS_PROMPT, combined_info=combined_info)
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

@function_tool
async def judge_answer(inputs: JudgeInput) -> JudgeOutput:
    prompt = load_prompt(
                    "judge_answer_prompt.j2",
                    task_description=inputs.task_description,
                    issue=inputs.issue,
                    other_input=inputs.other_input,
                    model_output=inputs.model_output)

    llm_response = await ask_gemini(prompt) 
    # Parse response
    lines = llm_response.strip().split('\n')
    correct_flag = None
    feedback = ""
    for line in lines:
        if line.lower().startswith("correct:"):
            correct_flag = line.split(':', 1)[1].strip().upper() == "YES"
        if line.lower().startswith("feedback:"):
            feedback = line.split(':', 1).strip()
    if correct_flag is None:
        correct_flag = False
        feedback = "LLM did not format answer properly."
    return JudgeOutput(is_correct=correct_flag, feedback=feedback)

judge_agent = Agent(
    name="JudgeAgent",
    instructions="You evaluate if a given model output, based on task, is correct for the input issue.",
    tools=[judge_answer],
    model=gemini_model
)

worker_agent = Agent(name="IssueWorkerAgent",
    instructions=load_prompt(WORKER_AGENT_INSTRUCTIONS),
    tools=[recommend_solutions],
    model=gemini_model)

planner_agent = Agent(
    name="IssuePlannerAgent",
    instructions=load_prompt(REACT_INSTRUCTIONS),
    tools=[
        worker_agent.as_tool(
            tool_name="analyze_issue",
            tool_description="Analyze a banking issue dictionary and recommend remediation suggestions. Give me in a single paragraph."
        ),
        judge_agent.as_tool(
            tool_name="judge_solution",
            tool_description="Evaluate if the answer given by an agent is correct for a given issue and task."
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