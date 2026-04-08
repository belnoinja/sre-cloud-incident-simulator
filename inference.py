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

# Configuration
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK") or "easy"
BENCHMARK = "cloud_env"
ENV_URL = os.getenv("OPENENV_BASE_URL") or "https://belnoinja-cloud-incident-simulator.hf.space"

MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.1
MAX_TOTAL_REWARD = 1.0

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main() -> None:
    print("[BOOT] Starting inference script", flush=True)
    
    # Initialize variables for the finally block
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = CloudEnvClient(base_url=ENV_URL)

        print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        # 1. Robust Environment Reset (Handles Cold Starts)
        result = None
        for attempt in range(1, 11):
            try:
                print(f"[INIT] Attempting environment reset ({attempt}/10)...", flush=True)
                result = await env.reset(task=TASK_NAME)
                if result and result.observation:
                    break
            except Exception as e:
                print(f"[DEBUG] Reset failed: {e}", flush=True)
                await asyncio.sleep(5)

        if not result or not result.observation:
            print("[FATAL] Environment unreachable after 10 attempts.", flush=True)
            return

        obs = result.observation
        messages = [
            {"role": "system", "content": "You are a professional SRE. Investigate and resolve cloud issues using the execute_cloud_command tool."},
            {"role": "user", "content": f"The environment is ready. Initial State: {obs.message}"}
        ]

        # 2. Agent Loop
        for step in range(1, MAX_STEPS + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "execute_cloud_command",
                            "description": "Executes a cloud command.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                    "args": {"type": "object"}
                                },
                                "required": ["command", "args"]
                            }
                        }
                    }],
                    tool_choice="auto",
                    temperature=0.1
                )

                msg = response.choices[0].message
                messages.append(msg) # Must add assistant message to history

                if not msg.tool_calls:
                    print(f"[DEBUG] Step {step}: Model provided final answer.", flush=True)
                    break

                # Process the first tool call
                tool_call = msg.tool_calls[0]
                args_json = json.loads(tool_call.function.arguments)
                cmd = args_json.get("command")
                args = args_json.get("args", {})

                # Execute in environment
                step_res = await env.step(CloudEnvAction(command=cmd, args=args))
                
                # Log metrics
                rew = float(step_res.reward or 0.0)
                rewards.append(rew)
                steps_taken = step
                log_step(step, f"{cmd}({args})", rew, step_res.done, step_res.observation.error)

                # CRITICAL: Always provide tool output back to the model
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": step_res.observation.error if step_res.observation.error else step_res.observation.output
                })

                if step_res.done:
                    print("[DEBUG] Environment signaled completion.", flush=True)
                    break

            except Exception as e:
                print(f"[ERROR] Step {step} failed: {e}", flush=True)
                break

        # 3. Final Scoring
        total_reward = sum(rewards)
        score = min(max(total_reward / MAX_TOTAL_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[FATAL ERROR] Main loop crashed: {e}", flush=True)
        traceback.print_exc()
    
    finally:
        # 4. Mandatory Final Logging & Shutdown
        try:
            await env.close()
        except:
            pass
        log_end(success, steps_taken, score, rewards)
        sys.exit(0) # Exit with 0 to ensure logs are processed

if __name__ == "__main__":
    asyncio.run(main())