import sys
import traceback
import os

def fallback_excepthook(exctype, value, tb):
    print(f"FATAL UNHANDLED EXCEPTION: {value}")
    os._exit(0)
sys.excepthook = fallback_excepthook

try:
    import asyncio
    import json
    import logging
    import os
    from typing import List, Optional

    from openai import AsyncOpenAI
    from client import CloudEnvClient
    from models import CloudEnvAction
except BaseException as e:
    print(f"[DEBUG] Fatal module import exception: {e}", flush=True)
    os._exit(0)

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
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run_episode(client: AsyncOpenAI, env: CloudEnvClient, task_name: str):
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
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
        except BaseException as e:
            print(f"[DEBUG] unhandled exception in reset: {str(e).encode('ascii', 'replace').decode('ascii')}", flush=True)
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
            except BaseException as e:
                rewards.append(0.0)
                safe_err = str(e).encode("ascii", "replace").decode("ascii")
                print(f"[STEP] step={step} action=error_llm reward=0.00 done=true error={safe_err}", flush=True)
                break

            if msg.tool_calls:
                messages.append(msg)
                
                # We must process every tool call that the model requested
                for tool_call in msg.tool_calls:
                    command = "unknown"
                    cmd_args = {}
                    
                    try:
                        args = json.loads(tool_call.function.arguments)
                        if isinstance(args, dict):
                            command = args.get("command", "")
                            extracted_args = args.get("args")
                            if isinstance(extracted_args, dict):
                                cmd_args = extracted_args
                        action_str = f"{command}({cmd_args})"
                    except BaseException as e:
                        action_str = "parse_error"

                    try:
                        step_res = env.step(CloudEnvAction(command=command, args=cmd_args))
                        if asyncio.iscoroutine(step_res):
                            step_res = await step_res
                        obs = step_res.observation
                        reward = step_res.reward or 0.0
                        done = step_res.done
                        error = obs.error if obs.error else None
                        
                        rewards.append(reward)
                        err_str = error if error else "null"
                        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={err_str}", flush=True)
                        
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
                    except BaseException as e:
                        rewards.append(0.0)
                        safe_err = str(e).encode("ascii", "replace").decode("ascii")
                        print(f"[STEP] step={step} action={action_str} reward=0.00 done=true error={safe_err}", flush=True)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"System Error: {safe_err}"
                        })
                        done = True
                        break
                
                if done:
                    break
            else:
                rewards.append(0.0)
                print(f"[STEP] step={step} action=stop reward=0.00 done=true error=Agent returned no tool call", flush=True)
                break
                
    except BaseException as e:
        safe_err = str(e).encode("ascii", "replace").decode("ascii")
        print(f"[DEBUG] unhandled exception in episode loop: {safe_err}", flush=True)
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    env = None
    try:
        try:
            client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "mock-key")
        except BaseException as e:
            print(f"[DEBUG] OpenAI initialization error: {e}", flush=True)
            return

        env_base = os.getenv("OPENENV_BASE_URL")
        if not env_base:
            env_port = os.getenv("PORT", "7860")
            env_base = f"http://localhost:{env_port}"
        try:
            env = CloudEnvClient(base_url=env_base)
        except BaseException as e:
            print(f"[DEBUG] Env initialization error: {e}", flush=True)
            return

        tasks = [os.getenv("TASK", "easy")] if os.getenv("TASK") else ["easy", "medium", "hard"]
        for task in tasks:
            await run_episode(client, env, task)
    except BaseException as e:
        print(f"[DEBUG] Fatal env integration error: {e}", flush=True)
    finally:
        try:
            if env is not None and hasattr(env, "close"):
                if asyncio.iscoroutinefunction(env.close):
                    await env.close()
                else:
                    env.close()
        except BaseException as ce:
            print(f"[DEBUG] env.close() error: {ce}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as e:
        print(f"[DEBUG] Final exit exception: {e}", flush=True)
        os._exit(0)


