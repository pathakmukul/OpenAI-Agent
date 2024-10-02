import os
from openai import OpenAI
import time
import json
import matplotlib.pyplot as plt
import io
import base64
from config import OPENAI_API_KEY

def run_assistant(context):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    assistant = client.beta.assistants.create(
        instructions=(
            "You are a data analyst reporter. Generate a comprehensive report on the analysis, "
            "including key findings, interpretation of results, and recommendations."
        ),
        model="gpt-4o-mini",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "generate_visualization",
                    "description": "Generate a visualization based on the query result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chart_type": {
                                "type": "string",
                                "enum": ["bar", "line", "scatter", "pie"],
                                "description": "The type of chart to generate"
                            },
                            "data": {
                                "type": "object",
                                "description": "The data to visualize"
                            }
                        },
                        "required": ["chart_type", "data"]
                    }
                }
            }
        ]
    )

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Generate a report based on this context: {json.dumps(context)}"
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
            if tool_call.function.name == "generate_visualization":
                arguments = json.loads(tool_call.function.arguments)
                plt.figure(figsize=(10, 6))
                if arguments["chart_type"] == "bar":
                    plt.bar(arguments["data"].keys(), arguments["data"].values())
                elif arguments["chart_type"] == "line":
                    plt.plot(list(arguments["data"].values()))
                elif arguments["chart_type"] == "scatter":
                    plt.scatter(range(len(arguments["data"])), list(arguments["data"].values()))
                elif arguments["chart_type"] == "pie":
                    plt.pie(arguments["data"].values(), labels=arguments["data"].keys())
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                output = f"<img src='data:image/png;base64,{image_base64}' alt='Visualization'>"
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": output
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
    report = ''
    for message in messages.data:
        if message.role == "assistant":
            if hasattr(message, 'content') and message.content:
                for content_item in message.content:
                    if hasattr(content_item, 'text'):
                        report += content_item.text.value
                    elif hasattr(content_item, 'image_file'):
                        report += f"\n\n{content_item.image_file.file_id}\n\n"
    return report

# Example usage
if __name__ == "__main__":
    sample_context = {
        "query_result": {"A": 10, "B": 20, "C": 15},
        "dataframes": {"sample_data": {"shape": [3, 3]}}
    }
    result = run_assistant(sample_context)
    print(result)