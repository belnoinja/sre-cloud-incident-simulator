import os
import json
import logging
from openai import OpenAI
from client import CloudEnvClient
from models import CloudEnvAction

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Tools definition for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_cloud_command",
            "description": "Executes a command on the simulated cloud infrastructure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run. Valid options: describe_volumes, delete_volume, describe_security_groups, update_security_group, describe_instances, read_logs, modify_instance_attribute, start_instance"
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments for the command. Example: {'volume_id': 'vol-002'} or {'sg_id': 'sg-db01', 'port': 5432, 'cidr': '10.0.0.0/8'} or {'instance_id': 'i-worker01'}"
                    }
                },
                "required": ["command", "args"]
            }
        }
    }
]

def run_task(agent_client, env, task_name: str) -> float:
    print(f"\n{'='*50}\nStarting Task: {task_name.upper()}\n{'='*50}")
    
    # 1. Reset Environment
    result = env.reset(task=task_name)
    obs = result.observation
    
    print(f"Task Objective: {obs.message}")
    
    messages = [
        {"role": "system", "content": "You are an SRE on a simulated cloud system. You must resolve the task described in the initialization message by taking deliberate execute_cloud_command actions. You have strict bounds, do not guess volume IDs or instance types without verifying them."},
        {"role": "user", "content": f"The environment has initialized. Your task:\n{obs.message}"}
    ]
    
    max_turns = 15
    for turn in range(max_turns):
        print(f"\n--- Turn {turn+1} ---")
        
        # 2. Call OpenAI Model
        response = agent_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        
        # Determine if model returned a tool call
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            command = args.get("command")
            cmd_args = args.get("args", {})
            print(f"Agent Action: {command}({cmd_args})")
            
            messages.append(msg) # Append AI message
            
            # 3. Step Environment
            action = CloudEnvAction(command=command, args=cmd_args)
            step_res = env.step(action)
            obs = step_res.observation
            
            # Print output
            outpreview = obs.output[:200] + ('...' if len(obs.output) > 200 else '')
            if obs.error:
                print(f"Environment Error: {obs.error}")
                feedb = f"Error: {obs.error}"
            else:
                print(f"Environment Output:\n{outpreview}")
                feedb = f"Output: {obs.output}"
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": feedb
            })
            
            if obs.done:
                print(f"\n=> Task Finished! Reward: {obs.reward} | Terminal Message: {obs.message}")
                return obs.reward
        else:
            print("Agent stopped answering with tool calls.")
            break
            
    print("\n=> Failed (Max turns reached).")
    return 0.0

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Set OPENAI_API_KEY to run baseline.")
        exit(1)
        
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    tasks = ["easy", "medium", "hard"]
    
    # We must start the fastapi server locally to test against HTTP using EnvClient 
    # Or just run it inside a mock thread.
    # The simplest baseline uses the synchronous EnvClient wrapper over the URL:
    
    try:
        # Note: server must be running on API_BASE_URL (uv run server or uvicorn server.app:app)
        with CloudEnvClient(base_url=API_BASE_URL).sync() as env:
            for task in tasks:
                reward = run_task(client, env, task)
                print(f"Task '{task}' achieved reward: {reward}")
    except Exception as e:
        print(f"Could not connect to {API_BASE_URL}. Ensure the server is running.")
        print(f"Error: {e}")
