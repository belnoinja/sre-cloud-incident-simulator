import asyncio
import json
import os
import sys
import traceback
import textwrap
from typing import List, Optional
from dotenv import load_dotenv

from openai import OpenAI
from client import CloudEnvClient
from models import CloudEnvAction

# Load environment variables
load_dotenv()

# Environment Configuration
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK") or "easy"
BENCHMARK = "cloud_env"
ENV_URL = os.getenv("OPENENV_BASE_URL") or "https://belnoinja-cloud-incident-simulator.hf.space"

MAX_STEPS = 15
TEMPERATURE = 0.1
MAX_TOKENS = 1000
SUCCESS_SCORE_THRESHOLD = 0.1

# Max possible reward for the task (usually 1.0 for cloud success)
MAX_TOTAL_REWARD = 1.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a professional SRE agent specialized in cloud infrastructure. 
    Your goal is to investigate and resolve all issues in the simulated environment. 
    Start by exploring resources (Describe) and then delete or modify them as needed. 
    Use the execute_cloud_command tool to take action.
    """
).strip()

tools = [{
    "type": "function",
    "function": {
        "name": "execute_cloud_command",
        "description": "Executes a command on the simulated cloud infrastructure.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Supported commands: describe_volumes, delete_volume, describe_security_groups, update_security_group, describe_instances, read_logs, modify_instance_attribute, start_instance"
                },
                "args": {"type": "object"}
            },
            "required": ["command", "args"]
        }
    }
}]

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

def get_model_response(client: OpenAI, messages: List[dict]):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return None

async def wait_for_env(env: CloudEnvClient, retries=60):
    for i in range(retries):
        try:
            res = await env.reset(task=TASK_NAME)
            if res and res.observation:
                return res
        except Exception:
            await asyncio.sleep(2)
    return None

async def main() -> None:
    # 🔥 BOOT LOG
    print("[BOOT] script loaded", flush=True)
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CloudEnvClient(base_url=ENV_URL)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Initial Reset with Cold Start Handling
        result = await wait_for_env(env)
        if not result or not result.observation:
            print("[DEBUG] Failed to initialize environment", flush=True)
            return

        obs = result.observation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"The environment is ready.\n\nInitial State:\n{obs.message}"}
        ]

        for step in range(1, MAX_STEPS + 1):
            msg = get_model_response(client, messages)
            if not msg:
                break
            
            if not msg.tool_calls:
                # Agent stopped acting
                break

            messages.append(msg)
            
            # For simplicity, we execute only the first tool call per step if multiple exist
            tool_call = msg.tool_calls[0]
            try:
                args = json.loads(tool_call.function.arguments)
                command = args.get("command")
                cmd_args = args.get("args", {})
                action_str = f"{command}({cmd_args})"
            except Exception as e:
                log_step(step=step, action="parse_error", reward=0.00, done=False, error=str(e))
                break

            try:
                step_res = await env.step(CloudEnvAction(command=command, args=cmd_args))
                obs = step_res.observation
                reward = float(step_res.reward or 0.0)
                done = step_res.done
                error = obs.error

                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: {obs.error}" if obs.error else f"Output: {obs.output}"
                })

                if done:
                    break

            except Exception as e:
                log_step(step=step, action=action_str, reward=0.00, done=True, error=str(e))
                break

        # Calculate final score and success
        total_reward = sum(rewards)
        score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal execution error: {e}", flush=True)
        traceback.print_exc()

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        # Ensure we exit with 0 to allow log capture
        sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        # Final safety catch for startup errors
        print(f"[DEBUG] Startup crash: {e}", flush=True)
        sys.exit(0)