import asyncio
import json
import os
import sys
import traceback
from typing import List

try:
    from openai import AsyncOpenAI
    from client import CloudEnvClient
    from models import CloudEnvAction
except ImportError:
    print("[FATAL] Import failed", flush=True)
    traceback.print_exc()
    sys.exit(1)


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "cloud_env"
MAX_STEPS = 15


tools = [{
    "type": "function",
    "function": {
        "name": "execute_cloud_command",
        "description": "Execute cloud infra command",
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


async def run_episode(client, env, task_name):
    print(f"[START] {task_name}", flush=True)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        # ✅ RETRY RESET
        for attempt in range(3):
            try:
                res = await env.reset(task=task_name)
                if res and res.observation:
                    break
            except Exception:
                print(f"[WARN] reset retry {attempt+1}", flush=True)
                await asyncio.sleep(2)
        else:
            raise RuntimeError("Env reset failed after retries")

        obs = res.observation

        messages = [
            {"role": "system", "content": "You are an SRE. Use tools."},
            {"role": "user", "content": obs.message}
        ]

        for step in range(1, MAX_STEPS + 1):
            steps_taken = step

            # ✅ LLM CALL
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1
                )
                msg = response.choices[0].message
            except Exception:
                print("[FATAL] LLM failed", flush=True)
                traceback.print_exc()
                raise

            if not msg or not msg.tool_calls:
                print("[INFO] No tool calls, stopping", flush=True)
                break

            messages.append(msg)

            for tool_call in msg.tool_calls:

                # ✅ SAFE PARSE
                try:
                    if not tool_call.function:
                        raise ValueError("Missing function in tool call")

                    args = json.loads(tool_call.function.arguments or "{}")
                    command = args.get("command")
                    cmd_args = args.get("args", {})

                    if not command:
                        raise ValueError("Missing command")

                except Exception as e:
                    err = f"Parse error: {e}"
                    print(f"[ERROR] {err}", flush=True)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err
                    })
                    continue

                # ✅ ENV STEP
                try:
                    step_res = await env.step(
                        CloudEnvAction(command=command, args=cmd_args)
                    )

                    obs = step_res.observation
                    reward = float(step_res.reward or 0.0)
                    done = step_res.done

                    rewards.append(reward)

                    print(f"[STEP] {step} {command} reward={reward}", flush=True)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": obs.output or obs.error or "OK"
                    })

                    if done:
                        success = reward > 0
                        score = reward
                        return

                except Exception:
                    print("[ERROR] env.step failed", flush=True)
                    traceback.print_exc()

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Execution failed"
                    })

    except Exception:
        print("[FATAL] Episode crashed", flush=True)
        traceback.print_exc()
        raise

    finally:
        print(f"[END] success={success} steps={steps_taken}", flush=True)


async def main():
    print("[DEBUG] Start", flush=True)

    env = None

    try:
        client = AsyncOpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or "mock-key"
        )

        env_base = os.getenv("OPENENV_BASE_URL") or f"http://localhost:{os.getenv('PORT','7860')}"

        print(f"[DEBUG] env={env_base}", flush=True)

        env = CloudEnvClient(base_url=env_base)

        tasks = [os.getenv("TASK")] if os.getenv("TASK") else ["easy", "medium", "hard"]

        for t in tasks:
            await run_episode(client, env, t)

    except Exception:
        print("[FATAL] main crashed", flush=True)
        traceback.print_exc()
        sys.exit(1)

    finally:
        if env:
            try:
                await env.close()
            except:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        print("[FATAL] script crash", flush=True)
        traceback.print_exc()
        sys.exit(1)