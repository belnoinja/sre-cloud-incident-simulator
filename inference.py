import asyncio
import json
import logging
import os
import sys
import traceback
from typing import List

try:
    from openai import AsyncOpenAI
    from client import CloudEnvClient
    from models import CloudEnvAction
except ImportError as e:
    print("[DEBUG] Critical Import Error:", flush=True)
    traceback.print_exc()
    sys.exit(1)  # ❗ fail properly


# Constants
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "cloud_env"
MAX_STEPS = 15


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
                        "description": "Valid options: describe_volumes, delete_volume, describe_security_groups, update_security_group, describe_instances, read_logs, modify_instance_attribute, start_instance"
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments for the command"
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
        # RESET with Retry
        res = None
        for attempt in range(3):
            try:
                res = await env.reset(task=task_name)
                if res and res.observation:
                    break
            except Exception as e:
                if attempt == 2:
                    print(f"[INTERNAL ERROR] Failed to reset env after 3 attempts: {e}", flush=True)
                    return
                print(f"[DEBUG] Env reset attempt {attempt+1} failed, retrying in 2s...", flush=True)
                await asyncio.sleep(2)
        
        if not res or not res.observation:
            print("[INTERNAL ERROR] Reset response empty or invalid", flush=True)
            return

        obs = res.observation

        messages = [
            {
                "role": "system",
                "content": "You are an SRE on a simulated cloud system. Take precise actions."
            },
            {
                "role": "user",
                "content": f"The environment has initialized:\n{obs.message}"
            }
        ]

        # LOOP
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
                if not response or not response.choices:
                    print(f"[ERROR] LLM returned empty response choices at step {step}", flush=True)
                    break
                    
                msg = response.choices[0].message
                if not msg:
                    print(f"[ERROR] LLM returned empty message at step {step}", flush=True)
                    break

            except Exception as e:
                print(f"[ERROR] LLM call failed at step {step}: {e}", flush=True)
                traceback.print_exc()
                break # Non-recoverable call failure

            if not msg.tool_calls:
                print(f"[STEP] step={step} No tool call (stop/final message)", flush=True)
                break

            messages.append(msg)

            for tool_call in msg.tool_calls:
                # 1. Parse Args with Error Capture
                error_content = None
                command = "unknown"
                cmd_args = {}
                
                try:
                    args = json.loads(tool_call.function.arguments)
                    command = args.get("command")
                    cmd_args = args.get("args", {})
                    if not command:
                        error_content = "Command name missing in tool arguments"
                except Exception as e:
                    error_content = f"JSON Parsing Error: {str(e)}. Ensure arguments are valid JSON."

                # 2. Execute Step with Recovery
                if not error_content:
                    action_str = f"{command}({cmd_args})"
                    try:
                        step_res = await env.step(
                            CloudEnvAction(command=command, args=cmd_args)
                        )

                        obs = step_res.observation
                        reward = float(step_res.reward or 0.0)
                        done = step_res.done
                        rewards.append(reward)

                        print(
                            f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done}",
                            flush=True
                        )

                        if done:
                            score = max(0.0, min(1.0, reward))
                            success = score > 0
                            # Append success/final obs before returning
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Final Observation: {obs.message}"
                            })
                            return

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {obs.error}" if obs.error else f"Output: {obs.output}"
                        })

                    except Exception as e:
                        print(f"[ERROR] Env step request failed: {e}", flush=True)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Network Error: Connection failed or timed out during {command}."
                        })
                else:
                    # Report Parsing/Command error back to LLM
                    print(f"[ERROR] Tool call validation failed: {error_content}", flush=True)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {error_content}"
                    })

    except Exception as e:
        print(f"[INTERNAL ERROR] Episode crashed fatally: {e}", flush=True)
        traceback.print_exc()

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={success} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)


async def main():
    print("[DEBUG] Inference script initialization", flush=True)
    env = None

    try:
        # INIT
        client = AsyncOpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or "mock-key"
        )

        env_base = os.getenv("OPENENV_BASE_URL")
        if not env_base:
            env_port = os.getenv("PORT", "7860")
            env_base = f"http://localhost:{env_port}"

        print(f"[DEBUG] Environment endpoint set to {env_base}", flush=True)
        env = CloudEnvClient(base_url=env_base)

        tasks = [os.getenv("TASK")] if os.getenv("TASK") else ["easy", "medium", "hard"]

        for task in tasks:
            await run_episode(client, env, task)

    except Exception as e:
        print(f"[FATAL ERROR] Setup failed: {e}", flush=True)
        traceback.print_exc()
        # Only exit with 1 if the script couldn't even reach the task loop
        sys.exit(1)

    finally:
        if env:
            try:
                await env.close()
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as e:
        print(f"[DEBUG] Script exit triggered: {type(e).__name__}", flush=True)
        if not isinstance(e, SystemExit):
            traceback.print_exc()
            sys.exit(1)
        else:
            sys.exit(e.code)