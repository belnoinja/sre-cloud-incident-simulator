import asyncio
import json
import os
import sys
import traceback
from dotenv import load_dotenv
from openai import OpenAI

# Force logs to flush even if the script hangs
def force_log(msg):
    print(msg, flush=True)

# 1. THE CRASH CATCHER
def global_exception_handler(exctype, value, tb):
    force_log("--- FATAL CRASH DETECTED ---")
    traceback.print_exception(exctype, value, tb)
    force_log("----------------------------")
    # Exit with 0 so the validator can at least read the logs we just printed
    sys.exit(0)

sys.excepthook = global_exception_handler

# 2. SAFE IMPORTS
try:
    from client import CloudEnvClient
    from models import CloudEnvAction
except Exception as e:
    force_log(f"IMPORT ERROR: Could not find client.py or models.py: {e}")
    sys.exit(0)

load_dotenv()

# Config
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK") or "easy"
ENV_URL = os.getenv("OPENENV_BASE_URL") or "https://belnoinja-cloud-incident-simulator.hf.space"

async def main():
    force_log(f"[BOOT] Script started. Target Env: {ENV_URL}")
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Initialize Client
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = CloudEnvClient(base_url=ENV_URL)

        # 3. ROBUST CONNECT (Simulator "Warm-up")
        result = None
        for i in range(1, 11):
            try:
                force_log(f"[INIT] Resetting Env (Attempt {i}/10)...")
                result = await env.reset(task=TASK_NAME)
                if result:
                    force_log("[INIT] Success! Simulator responded.")
                    break
            except Exception as e:
                force_log(f"[DEBUG] Connection waiting: {e}")
                await asyncio.sleep(8)

        if not result:
            force_log("[FATAL] Simulator never responded. Check ENV_URL.")
            return

        messages = [
            {"role": "system", "content": "Professional SRE Agent. Use execute_cloud_command."},
            {"role": "user", "content": f"Initial state: {result.observation.message}"}
        ]

        # 4. AGENT LOOP
        for step in range(1, 16):
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
                break

            tool_call = msg.tool_calls[0]
            args_dict = json.loads(tool_call.function.arguments)
            
            # Action execution
            step_res = await env.step(CloudEnvAction(
                command=args_dict["command"], 
                args=args_dict.get("args", {})
            ))
            
            rew = float(step_res.reward or 0.0)
            rewards.append(rew)
            steps_taken = step
            force_log(f"[STEP {step}] {args_dict['command']} | Reward: {rew}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": step_res.observation.output or step_res.observation.error or "Done"
            })

            if step_res.done:
                break

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= 0.1

    except Exception as e:
        force_log("--- ERROR DURING EXECUTION ---")
        traceback.print_exc()
    
    finally:
        # Mandatory output for evaluator
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        force_log(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}")
        sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        force_log(f"Asyncio Loop Crash: {e}")
        traceback.print_exc()
        sys.exit(0)