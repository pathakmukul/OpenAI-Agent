import os
from openai import OpenAI
import time
import json
from config import OPENAI_API_KEY
def run_assistant(prompt, context):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    assistant = client.beta.assistants.create(
        instructions=(
            "You are a data analyst validator. Review the user's original prompt, examine the context "
            "containing data information and query results, and validate if the executed query satisfies "
            "the user's request."
        ),
        model="gpt-4o-mini",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "validate_result",
                    "description": "Validate if the query result satisfies the user's request",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_valid": {
                                "type": "boolean",
                                "description": "Whether the result is valid or not"
                            },
                            "message": {
                                "type": "string",
                                "description": "Explanation of the validation result"
                            }
                        },
                        "required": ["is_valid", "message"]
                    }
                }
            }
        ]
    )

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Original prompt: {prompt}\n\nContext: {json.dumps(context)}"
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
            if tool_call.function.name == "validate_result":
                arguments = json.loads(tool_call.function.arguments)
                output = arguments
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
    validation_result = ''
    for message in messages.data:
        if message.role == "assistant":
            if hasattr(message, 'content') and message.content:
                for content_item in message.content:
                    if hasattr(content_item, 'text'):
                        validation_result += content_item.text.value
                        break
    return validation_result

# Example usage
if __name__ == "__main__":
    sample_context = {
        "query_result": 3,
        "dataframes": {"sample_data": {"shape": [3, 2]}}
    }
    result = run_assistant("Count the number of rows in the sample_data DataFrame", sample_context)
    print(result)