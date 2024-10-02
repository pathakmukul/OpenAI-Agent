import os
from dotenv import load_dotenv
from openai import OpenAI
import time
from config import OPENAI_API_KEY
load_dotenv(override=True)

def run_assistant(prompt, execute_code=False):
	# Set API key directly
	client = OpenAI(api_key=OPENAI_API_KEY)

	# Adjust assistant instructions to emphasize using provided information
	assistant_instructions = (
		"You are an expert programmer. Use all provided information, such as image URLs or other data, to write code that accomplishes the task."
	)  # Fixed string termination

	# Choose tools based on whether we need to execute code
	tools = [{"type": "code_interpreter"}] if execute_code else []

	# Create an assistant with the injected prompt
	assistant = client.beta.assistants.create(
		name="Code Generator",
		instructions=assistant_instructions,
		tools=tools,
		model="gpt-4o"
	)

	# Create a Thread
	thread = client.beta.threads.create()

	# Add the user's prompt to the thread
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

	# Wait for the run to complete
	while True:
		run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
		if run_status.status in ['completed', 'failed', 'requires_action']:
			break
		time.sleep(1)

	# Retrieve and return the assistant's response
	messages = client.beta.threads.messages.list(thread_id=thread.id)
	code_output = ''  # Fixed initialization
	for message in messages.data:
		if message.role == "assistant":
			if hasattr(message, 'content') and message.content:
				for content_item in message.content:
					if hasattr(content_item, 'text'):
						code_output += content_item.text.value
					elif execute_code and hasattr(content_item, 'execution_output'):
						code_output += f"\nExecution Output: \n{content_item.execution_output}\n"  # Fixed string formatting
				break
	return code_output