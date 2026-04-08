import asyncio
import json
import os
import sys
import traceback
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

# Safe imports for custom modules
try:
    from client import CloudEnvClient
    from models import CloudEnvAction
except ImportError as e:
    print(f"[BOOT ERROR] Local files missing: {e}", flush=True)
    sys.exit(0)

load_dotenv()

# Config
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK") or "easy"
ENV_URL = os.getenv("OPENENV_BASE_URL") or "https://belnoinja-cloud-incident-simulator.hf.space"

async def main():
    print(f"[BOOT] Starting Inference. Model: {MODEL_NAME}, Task: {TASK_NAME}", flush=True)
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        if not API_KEY:
            print("[ERROR] API_KEY/HF_TOKEN is missing.", flush=True)
            return

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = CloudEnvClient(base_url=ENV_URL)

        # --- 1. WAIT FOR SIMULATOR ---
        result = None
        for i in range(1, 13): # 12 attempts * 10s = 120s max wait
            try:
                print(f"[INIT] Resetting Env (Attempt {i}/12)...", flush=True)
                result = await env.reset(task=TASK_NAME)
                if result and result.observation:
                    print("[INIT] Connection Successful.", flush=True)
                    break
            except Exception as e:
                print(f"[DEBUG] Env pending: {e}", flush=True)
                await asyncio.sleep(10)

        if not result:
            print("[FATAL] Environment failed to respond.", flush=True)
            return

        messages = [
            {"role": "system", "content": "You are a professional SRE. Use execute_cloud_command to fix infrastructure issues."},
            {"role": "user", "content": f"Initial state: {result.observation.message}"}
        ]

        # --- 2. AGENT LOOP ---
        for step in range(1, 16):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "execute_cloud_command",
                            "description": "Execute SRE actions",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                    "args": {"type": "object"}
                                },
                                "required": ["command", "args"]
                            }
                        }
                    }]
                )

                msg = response.choices[0].message
                messages.append(msg)

                if not msg.tool_calls:
                    print(f"[STEP {step}] No more tool calls. Ending.", flush=True)
                    break

                tool_call = msg.tool_calls[0]
                args_dict = json.loads(tool_call.function.arguments)
                
                # EXECUTE
                step_res = await env.step(CloudEnvAction(
                    command=args_dict["command"], 
                    args=args_dict.get("args", {})
                ))
                
                rew = float(step_res.reward or 0.0)
                rewards.append(rew)
                steps_taken = step
                
                print(f"[STEP {step}] Action: {args_dict['command']} | Reward: {rew}", flush=True)

                # Append tool result (Required for API consistency)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": step_res.observation.output or step_res.observation.error or "Done"
                })

                if step_res.done:
                    print("[DEBUG] Environment signaled completion.", flush=True)
                    break

            except Exception as e:
                print(f"[STEP ERROR] Step {step} failed: {e}", flush=True)
                break

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= 0.1

    except Exception:
        print("[CRITICAL ERROR]")
        traceback.print_exc()
    
    finally:
        # Mandatory output format for the evaluator
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())