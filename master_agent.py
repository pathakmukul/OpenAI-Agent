import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.theme import Theme
from file_agent import run_assistant as run_file_agent
from image_agent import run_assistant as run_image_agent
from input_file import user_task
from data_loader_agent import run_data_loader
from query_builder_agent import run_assistant as run_query_builder
from data_analyst_validator_agent import run_assistant as run_validator
from data_analyst_reporter_agent import run_assistant as run_reporter
from config import OPENAI_API_KEY

load_dotenv(override=True)

# Initialize Rich console with a custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "spinner": "magenta",
    "code": "green",
    "image": "blue",
})
console = Console(theme=custom_theme)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def master_agent(user_task):
    console.print(f"[info]Master Agent received the task:[/info] '{user_task}'")

    # Use the model to decide which agents to invoke and determine dependencies
    planning_prompt = (
        f"<task>{user_task}</task>\n\n"
        f"<instructions>"
        f"As the Master Agent, you need to create a VERY PRECISE plan to complete the task by utilizing the following agents:\n"
        f"<agents>"
        f"<agent name='image'>Image Generation Agent: Generates images based on text prompts.</agent>\n"
        f"<agent name='code'>Code Generation Agent: Can write and execute code to accomplish programming tasks.</agent>\n"
        f"<agent name='file'>File Management Agent: Reads from and writes to files in the current directory, and can download images.</agent>\n"
        f"<agent name='data_loader'>Data Loader Agent: Loads and manages CSV files, creates DataFrames.</agent>\n"
        f"<agent name='query_builder'>Query Builder Agent: Interprets user prompts and builds queries for data analysis.</agent>\n"
        f"<agent name='validator'>Data Analyst Validator Agent: Validates query results against user requests.</agent>\n"
        f"<agent name='reporter'>Data Analyst Reporter Agent: Generates comprehensive reports on data analysis results.</agent>\n"
        f"</agents>\n"
        f"Determine which agents to use and the order in which to invoke them, based on dependencies.\n"
        f"Specify the plan in JSON format with the following structure:\n"
        f"<json_structure>"
        f"{{\n"
        f"  \"plan\": [\n"
        f"    {{ \"agent\": \"agent_name\", \"prompt\": \"prompt_for_agent\" }},\n"
        f"    ... \n"
        f"  ]\n"
        f"}}"
        f"</json_structure>\n"
        f"<rules>"
        f"- Remember, you can only use each agent once. If you need to use an agent more than once, you must include it in the plan. Image agents can make images, but not save them. File agents can save, read and download files, but not make images, etc.\n"
        f"- Only include the agents necessary for the task.\n"
        f"- Do not include any code blocks or additional text.\n"
        f"- Be VERY PRECISE with your plan and agent prompts; an agent prompt should have one instruction, and only one.\n"
        f"- DONT INCLUDE ```json or```, MUST BE VALID JSON.\n"
        f"</rules>"
        f"</instructions>"
    )

    console.print("[info]Master Agent is creating a plan to complete the task.[/info]")

    with console.status("[spinner]Planning...", spinner="dots") as status:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {
                    "role": "user",
                    "content": planning_prompt,
                }
            ]
        )
        
        plan_text = response.choices[0].message.content.strip()
        console.print(f"[info]Received plan: [/info]\n{plan_text}")

    # Parse the JSON plan
    try:
        plan = json.loads(plan_text)
        if 'plan' not in plan:
            raise ValueError("Plan does not contain 'plan' key.")
    except ValueError as e:
        console.print(f"[error]Error parsing plan: {e}[/error]")
        return

    # Execute the plan
    results = {}
    context = {}
    for step in plan['plan']:
        agent_name = step['agent']
        agent_prompt = step['prompt']
        console.print(f"[info]Invoking {agent_name.capitalize()} Agent...[/info]")
        
        # Include context in agent prompts
        if context:
            agent_prompt = f"{agent_prompt}\n\nContext from previous agents:\n{json.dumps(context, indent=2)}"
        
        if agent_name.lower() == 'image':
            with console.status("[spinner]Generating images...", spinner="dots") as status:
                image_result = run_image_agent(agent_prompt, client)  # Pass client as an argument
                console.print(f"[success]Image Generation Agent returned:[/success]\n{image_result}")
                results['image'] = image_result
                context['image'] = image_result

        elif agent_name.lower() == 'code':
            from code_agent import run_assistant as run_code_agent
            execute_code = step['prompt'].lower().startswith('execute')
            with console.status("[spinner]Generating or executing code...", spinner="dots") as status:
                code_result = run_code_agent(agent_prompt, execute_code=execute_code)
                console.print(f"[success]Code Generation Agent returned code: [/success]\n{code_result}")
                results['code'] = code_result
                
                # Extract code blocks (Python, HTML, etc.)
                code_blocks = re.findall(r'```(.*?)```', code_result, re.DOTALL)
                if code_blocks:
                    combined_code = '\n'.join(code_blocks)
                    results['code'] = combined_code
                    context['code'] = combined_code  # Store for context
                    console.print("[success]Code extracted and stored in context.[/success]")
                else:
                    console.print("[warning]No code blocks found in the generated result.[/warning]")
                    results['code'] = "No valid code was generated."
                    context['code'] = results['code']

        elif agent_name.lower() == 'file':
            with console.status("[spinner]Performing file operation...", spinner="dots") as status:
                file_result = run_file_agent(agent_prompt)
                console.print(f"[success]File Management Agent returned result:[/success]\n{file_result}")
                results['file'] = file_result
                context['file_result'] = file_result  # Store for context

        elif agent_name.lower() == 'data_loader':
            with console.status("[spinner]Loading data...", spinner="dots") as status:
                file_paths = agent_prompt.split(",")  # Assuming file paths are comma-separated
                context = run_data_loader(file_paths)
                console.print(f"[debug]Data Loader Agent returned context: {json.dumps(context, indent=2)}")
                console.print(f"[success]Data Loader Agent completed.[/success]")

        elif agent_name.lower() == 'query_builder':
            with console.status("[spinner]Building and executing query...", spinner="dots") as status:
                query_result = run_query_builder(agent_prompt, context)
                context['query_result'] = json.loads(query_result)
                console.print(f"[success]Query Builder Agent completed.[/success]")

        elif agent_name.lower() == 'validator':
            with console.status("[spinner]Validating results...", spinner="dots") as status:
                validation_result = run_validator(agent_prompt, context)
                context['validation_result'] = json.loads(validation_result)
                console.print(f"[success]Data Analyst Validator Agent completed.[/success]")
                if not context['validation_result']['is_valid']:
                    console.print(f"[error]{context['validation_result']['message']}[/error]")
                    return  # Stop execution if validation fails

        elif agent_name.lower() == 'reporter':
            with console.status("[spinner]Generating report...", spinner="dots") as status:
                report = run_reporter(context)
                results['report'] = report
                console.print(f"[success]Data Analyst Reporter Agent completed.[/success]")

        else:
            console.print(f"[warning]Unknown agent: {agent_name}[/warning]")

    # Combine and return the results
    final_response = ''

    if 'image' in results:
        image_panel = Panel(f"{results['image']}", title="Image Generation Result", border_style="image")
        final_response += f"{image_panel}\n\n"

    if 'code' in results:
        syntax = Syntax(results['code'], "html", theme="monokai", line_numbers=True)
        code_panel = Panel(syntax, title="Code Generation Result", border_style="code")
        final_response += f"{code_panel}\n\n"

    if 'file' in results:
        file_panel = Panel(f"{results['file']}", title="File Management Result", border_style="cyan")
        final_response += f"{file_panel}\n\n"

    if 'report' in results:
        report_panel = Panel(results['report'], title="Data Analysis Report", border_style="cyan")
        final_response += f"{report_panel}\n\n"

    console.print(final_response)
    return final_response

# Example usage
if __name__ == '__main__':
    # Example task
    user_task = "Analyze the data in sample_data.csv"
    
    # Example file paths
    file_paths = ["sample_data.csv"]
    
    # Load data
    context = run_data_loader(file_paths)
    console.print(f"[info]Data loaded with context: {json.dumps(context, indent=2)}")
    
    # Continue with the rest of the task
    response = master_agent(user_task)
    console.print(response)