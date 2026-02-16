
import asyncio
from agentic_framework.agents.agent1 import invoke_planner, PlannerInput, _cleanup_clients
from agentic_framework.tools.weaviate_kb import client
import json

async def main():
    sample_issue = {
        "issue_id": "12345",
        "issue_type": "Credit Card Fraud",
        "issue_description": "Customer reported unauthorized transactions on their credit card.",
        "resolution": "Card blocked and reissued, investigation completed."
    }
    
    # Create PlannerInput instance
    planner_input = PlannerInput(**sample_issue)
 
    # Step 1: Call on_invoke_tool with input -> returns coroutine object
    input_json = json.dumps({"input": planner_input.model_dump()})
    coroutine = invoke_planner.on_invoke_tool(None, input_json)  # pass ctx=None
    result = await coroutine

    print(result)
    print(result)
    client.close()
    await _cleanup_clients()

if __name__ == "__main__":
    asyncio.run(main())