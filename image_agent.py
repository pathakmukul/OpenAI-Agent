import os
import replicate
from dotenv import load_dotenv
from openai import OpenAI
import time
import json
from config import REPLICATE_API_TOKEN, OPENAI_API_KEY
load_dotenv(override=True)

def generate_image(user_prompt):
    # Ensure the API token is loaded from the .env file
    replicate_api_token = REPLICATE_API_TOKEN  # Direct assignment
    if not replicate_api_token:
        raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
    
    # Set up the client with the API token
    client = replicate.Client(api_token=replicate_api_token)  # Fixed client initialization

    # Run the model with the user's prompt
    output = client.run(
        "black-forest-labs/flux-dev",
        input={
            "prompt": user_prompt,
            "go_fast": True,
            "guidance": 3.5,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "png",
            "output_quality": 80,
            "prompt_strength": 0.8,
            "num_inference_steps": 28
        }
    )

    # The output is expected to be a list with a single URL
    if output and isinstance(output, list) and len(output) > 0:
        return output[0]
    else:
        raise ValueError("No image URL was generated")

def run_assistant(prompt, client):  # Pass client as an argument
    # Create an assistant with the injected prompt
    assistant = client.beta.assistants.create(
        instructions=(
            "You are an image generation assistant. Use the provided function to generate images based on user prompts."
        ),
        model="gpt-4o-mini",  # Fixed model name
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an image based on a user prompt",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_prompt": {
                                "type": "string",
                                "description": "The user's prompt for image generation"
                            }
                        },
                        "required": ["user_prompt"]
                    }
                }
            }
        ]
    )

    # Create a Thread
    thread = client.beta.threads.create()
    
    # Add the injected prompt to the Thread
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
            if tool_call.function.name == "generate_image":
                # Parse the arguments string as JSON
                arguments = json.loads(tool_call.function.arguments)
                user_prompt = arguments.get("user_prompt")
                image_url = generate_image(user_prompt)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": image_url
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
    image_url = None
    for message in messages.data:
        if message.role == "assistant":
            if hasattr(message, 'content') and message.content:
                for content_item in message.content:
                    if hasattr(content_item, 'text'):
                        image_url = content_item.text.value
                        break
    return image_url
