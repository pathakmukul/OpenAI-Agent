import os
from openai import OpenAI
import time
import json
import pandas as pd
from config import OPENAI_API_KEY

def run_assistant(prompt, context):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    assistant = client.beta.assistants.create(
        instructions=(
            "You are a query building assistant. Interpret user prompts for data analysis tasks "
            "and build Python queries based on the available data."
        ),
        model="gpt-4o-mini",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "execute_query",
                    "description": "Execute a Python query on a DataFrame",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The Python query to execute"
                            },
                            "df_name": {
                                "type": "string",
                                "description": "The name of the DataFrame to query"
                            }
                        },
                        "required": ["query", "df_name"]
                    }
                }
            }
        ]
    )

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Context: {json.dumps(context)}\n\nPrompt: {prompt}"
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status in ['completed', 'failed', 'requires_action']:
            break
        time.sleep(1)

    if run_status.status == 'requires_action':
        tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        for tool_call in tool_calls:
            if tool_call.function.name == "execute_query":
                arguments = json.loads(tool_call.function.arguments)
                df = globals()[arguments["df_name"]]
                result = eval(arguments["query"])
                output = result.to_dict() if isinstance(result, pd.DataFrame) else result
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps(output)
            })

        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == 'completed':
                break
            time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    query_result = ''
    for message in messages.data:
        if message.role == "assistant":
            if hasattr(message, 'content') and message.content:
                for content_item in message.content:
                    if hasattr(content_item, 'text'):
                        query_result += content_item.text.value
                        break
    return query_result

# Example usage
if __name__ == "__main__":
    sample_context = {
        "dataframes": {"sample_data": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})}
    }
    result = run_assistant("Count the number of rows in the sample_data DataFrame", sample_context)
    print(result)