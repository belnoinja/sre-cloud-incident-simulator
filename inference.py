import asyncio
import json
import logging
import os
import sys
import traceback
from typing import List, Optional

try:
    from openai import AsyncOpenAI
    from client import CloudEnvClient
    from models import CloudEnvAction
except ImportError as e:
    print(f"[DEBUG] Critical Import Error: {e}", flush=True)
    sys.exit(0)

# Constants
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "cloud_env"
MAX_STEPS = 15

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

async def run_episode(client: AsyncOpenAI, env: CloudEnvClient, task_name: str):
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        # 1. Reset
        try:
            res = await env.reset(task=task_name)
            obs = res.observation
        except Exception as e:
            print(f"[DEBUG] Reset failed: {e}", flush=True)
            return

        messages = [
            {"role": "system", "content": "You are an SRE on a simulated cloud system. You must resolve the task described in the initialization message by taking deliberate execute_cloud_command actions. You have strict bounds, do not guess volume IDs or instance types without verifying them."},
            {"role": "user", "content": f"The environment has initialized. Your task:\n{obs.message}"}
        ]
        
        # 2. Step Loop
        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1
                )
                msg = response.choices[0].message
            except Exception as e:
                print(f"[STEP] step={step} action=error_llm reward=0.00 done=true error={e}", flush=True)
                rewards.append(0.0)
                break

            if not msg.tool_calls:
                print(f"[STEP] step={step} action=stop reward=0.00 done=true error=Agent returned no tool call", flush=True)
                rewards.append(0.0)
                break

            messages.append(msg)
            
            # Process tool calls
            current_done = False
            for tool_call in msg.tool_calls:
                command = "unknown"
                cmd_args = {}
                
                try:
                    args = json.loads(tool_call.function.arguments)
                    command = args.get("command", "unknown")
                    cmd_args = args.get("args", {})
                except Exception:
                    pass
                
                action_str = f"{command}({cmd_args})"
                
                try:
                    step_res = await env.step(CloudEnvAction(command=command, args=cmd_args))
                    obs = step_res.observation
                    reward = float(step_res.reward or 0.0)
                    done = step_res.done
                    
                    rewards.append(reward)
                    err_str = obs.error if obs.error else "null"
                    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={err_str}", flush=True)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {obs.error}" if obs.error else f"Output: {obs.output}"
                    })
                    
                    if done:
                        score = max(0.0, min(1.0, float(reward)))
                        if score > 0: success = True
                        current_done = True
                        break
                except Exception as e:
                    print(f"[STEP] step={step} action={action_str} reward=0.00 done=true error={e}", flush=True)
                    rewards.append(0.0)
                    current_done = True
                    break
            
            if current_done:
                break
                
    except Exception as e:
        print(f"[DEBUG] Episode loop exception: {e}", flush=True)
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    print("[DEBUG] Inference script started", flush=True)
    env = None
    try:
        # Initialization
        try:
            client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "mock-key")
            
            env_base = os.getenv("OPENENV_BASE_URL")
            if not env_base:
                env_port = os.getenv("PORT", "7860")
                env_base = f"http://localhost:{env_port}"
            
            env = CloudEnvClient(base_url=env_base)
        except Exception as e:
            print(f"[DEBUG] Initialization failed: {e}", flush=True)
            return

        # Execution
        tasks = [os.getenv("TASK", "easy")] if os.getenv("TASK") else ["easy", "medium", "hard"]
        for task in tasks:
            await run_episode(client, env, task)

    except Exception as e:
        print(f"[DEBUG] Fatal error in main: {e}", flush=True)
    finally:
        if env:
            try:
                await env.close()
            except:
                pass
        print("[DEBUG] Inference script finished", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[DEBUG] Final script exception: {e}", flush=True)
    finally:
        sys.exit(0)


