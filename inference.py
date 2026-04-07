import sys
import traceback

def emergency_exit(e):
    print(f"[DEBUG] Fatal global exception: {e}", flush=True)
    sys.exit(0)

try:
    import asyncio
    import os
    import json
    import logging
    from typing import List, Optional

    from openai import AsyncOpenAI
    from client import CloudEnvClient
    from models import CloudEnvAction
except BaseException as e:
    emergency_exit(e)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
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

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run_episode(client: AsyncOpenAI, env: CloudEnvClient, task_name: str):
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        try:
            result = env.reset(task=task_name)
            if asyncio.iscoroutine(result):
                result = await result
            obs = result.observation
        except Exception as e:
            # Network issue or parsing issue in reset
            print(f"[DEBUG] unhandled exception in reset: {e}", flush=True)
            return

        messages = [
            {"role": "system", "content": "You are an SRE on a simulated cloud system. You must resolve the task described in the initialization message by taking deliberate execute_cloud_command actions. You have strict bounds, do not guess volume IDs or instance types without verifying them."},
            {"role": "user", "content": f"The environment has initialized. Your task:\n{obs.message}"}
        ]
        
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
                rewards.append(0.0)
                log_step(step=step, action="error_llm", reward=0.0, done=True, error=str(e))
                break

            action_str = "no_action"
            command = "unknown"
            cmd_args = {}
            
            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                try:
                    args = json.loads(tool_call.function.arguments)
                    command = args.get("command", "")
                    cmd_args = args.get("args", {})
                    action_str = f"{command}({cmd_args})"
                except Exception as e:
                    action_str = f"parse_error({e})"
                
                messages.append(msg)
                
                try:
                    step_res = env.step(CloudEnvAction(command=command, args=cmd_args))
                    if asyncio.iscoroutine(step_res):
                        step_res = await step_res
                    obs = step_res.observation
                    reward = step_res.reward or 0.0
                    done = step_res.done
                    error = obs.error if obs.error else None
                    
                    rewards.append(reward)
                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                    
                    feedb = f"Error: {obs.error}" if obs.error else f"Output: {obs.output}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": feedb
                    })
                    
                    if done:
                        score = max(0.0, min(1.0, float(reward)))
                        if score > 0:
                            success = True
                        break
                except Exception as e:
                    rewards.append(0.0)
                    log_step(step=step, action=action_str, reward=0.0, done=True, error=str(e))
                    break
            else:
                rewards.append(0.0)
                log_step(step=step, action="stop", reward=0.0, done=True, error="Agent returned no tool call")
                break
                
    except Exception as e:
        print(f"[DEBUG] unhandled exception in episode loop: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    env = None
    try:
        try:
            client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "mock-key")
        except Exception as e:
            print(f"[DEBUG] OpenAI initialization error: {e}", flush=True)
            return

        env_base = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
        try:
            env = CloudEnvClient(base_url=env_base)
        except Exception as e:
            print(f"[DEBUG] Env initialization error: {e}", flush=True)
            return

        tasks = [os.getenv("TASK", "easy")] if os.getenv("TASK") else ["easy", "medium", "hard"]
        for task in tasks:
            await run_episode(client, env, task)
    except Exception as e:
        print(f"[DEBUG] Fatal env integration error: {e}", flush=True)
    finally:
        try:
            if env is not None and hasattr(env, "close"):
                if asyncio.iscoroutinefunction(env.close):
                    await env.close()
                else:
                    env.close()
        except Exception as ce:
            print(f"[DEBUG] env.close() error: {ce}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as e:
        emergency_exit(e)

