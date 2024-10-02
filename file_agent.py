import os
from openai import OpenAI
import time
import json
import requests
from PIL import Image
from io import BytesIO
from config import OPENAI_API_KEY

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)  # Fixed the method call
        return f"File successfully written to {file_path}"  # Fixed the f-string
    except Exception as e:
        return f"Error writing file: {str(e)}"

def download_image(url, file_path):
    try:
        response = requests.get(url)  # Fixed variable name
        img = Image.open(BytesIO(response.content))
        img.save(file_path)
        return f"Image successfully downloaded and saved to {file_path}"  # Fixed the f-string
    except Exception as e:
        return f"Error downloading image: {str(e)}"

def run_assistant(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Create an assistant with file read/write capabilities
    assistant = client.beta.assistants.create(
        instructions=(
            "You are a file management assistant. Use the provided functions to read and write files in the current working directory, and download images."
        ),
        model="gpt-4o-mini",  # Fixed model name
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to be read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to be written"
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "download_image",
                    "description": "Download an image from a URL and save it to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the image to download"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "The path where the image should be saved"
                            }
                        },
                        "required": ["url", "file_path"]
                    }
                }
            }
        ]
    )

    # Create a Thread
    thread = client.beta.threads.create()  # Fixed parentheses

    # Add the user's prompt to the Thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )

    # Create a Run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Wait for the run to complete or require action
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status in ['completed', 'failed', 'requires_action']:
            break
        time.sleep(1)

    # If the run requires action, handle it
    if run_status.status == 'requires_action':
        tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        for tool_call in tool_calls:
            if tool_call.function.name in ["read_file", "write_file", "download_image"]:
                arguments = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "read_file":
                    output = read_file(arguments["file_path"])
                elif tool_call.function.name == "write_file":
                    output = write_file(arguments["file_path"], arguments["content"])
                else:  # download_image
                    output = download_image(arguments["url"], arguments["file_path"])
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": output
                })

        # Submit the tool outputs
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )

    # Wait for the run to complete after submitting tool outputs
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == 'completed':
            break
        time.sleep(1)

    # Retrieve and return the assistant's response
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    file_action_result = ''  # Fixed initialization
    for message in messages.data:
        if message.role == "assistant":
            if hasattr(message, 'content') and message.content:
                for content_item in message.content:
                    if hasattr(content_item, 'text'):
                        file_action_result += content_item.text.value
                        break
    return file_action_result
