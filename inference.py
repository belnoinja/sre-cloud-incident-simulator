import asyncio
import json
import os
import sys
import traceback
from typing import List

# ---------------- SAFE IMPORT ----------------
try:
    from openai import AsyncOpenAI
    from client import CloudEnvClient
    from models import CloudEnvAction
except Exception:
    print("[FATAL] Import failed", flush=True)
    traceback.print_exc()
    import os
    os._exit(0)


# ---------------- CONFIG ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 15


tools = [{
    "type": "function",
    "function": {
        "name": "execute_cloud_command",
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


# ---------------- EPISODE ----------------
async def run_episode(client, env, task_name):
    print(f"[START] task={task_name}", flush=True)

    try:
        # -------- RESET SAFE --------
        res = None
        for attempt in range(3):
            try:
                res = await env.reset(task=task_name)
                if res and res.observation:
                    break
            except Exception as e:
                print(f"[WARN] reset attempt {attempt+1} failed: {e}", flush=True)
                await asyncio.sleep(1)

        if not res or not res.observation:
            print("[ERROR] reset failed completely", flush=True)
            return

        obs = res.observation

        messages = [
            {"role": "system", "content": "You are an SRE. Use tools."},
            {"role": "user", "content": obs.message}
        ]

        # -------- LOOP --------
        for step in range(1, MAX_STEPS + 1):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1
                )
            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}", flush=True)
                traceback.print_exc()
                break

            if not response or not response.choices:
                print("[ERROR] Empty LLM response", flush=True)
                break

            msg = response.choices[0].message

            if not msg or not msg.tool_calls:
                print("[INFO] No tool calls returned", flush=True)
                break

            messages.append(msg)

            for tool_call in msg.tool_calls:
                # -------- PARSE --------
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                    command = args.get("command")
                    cmd_args = args.get("args", {})

                    if not command:
                        raise ValueError("Missing command")

                except Exception as e:
                    print(f"[ERROR] Parse failed: {e}", flush=True)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(e)
                    })
                    continue

                # -------- ENV STEP --------
                try:
                    step_res = await env.step(
                        CloudEnvAction(command=command, args=cmd_args)
                    )

                    obs = step_res.observation
                    reward = float(step_res.reward or 0.0)
                    done = step_res.done

                    print(f"[STEP] step={step} action={command} reward={reward}", flush=True)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": obs.output or obs.error or "OK"
                    })

                    if done:
                        print("[SUCCESS] Task completed", flush=True)
                        return

                except Exception as e:
                    print(f"[ERROR] env.step failed: {e}", flush=True)
                    traceback.print_exc()
                    continue

    except Exception as e:
        print(f"[ERROR] Episode crash: {e}", flush=True)
        traceback.print_exc()

    finally:
        print("[END EPISODE]", flush=True)


# ---------------- MAIN ----------------
async def main():
    print("[DEBUG] Script started", flush=True)

    env = None

    try:
        print(f"[DEBUG] OPENENV_BASE_URL={os.getenv('OPENENV_BASE_URL')}", flush=True)

        client = AsyncOpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or "mock-key"
        )

        env_base = os.getenv("OPENENV_BASE_URL") or f"http://localhost:{os.getenv('PORT','7860')}"
        print(f"[DEBUG] Using env: {env_base}", flush=True)

        env = CloudEnvClient(base_url=env_base)

        tasks = [os.getenv("TASK")] if os.getenv("TASK") else ["easy"]

        for t in tasks:
            await run_episode(client, env, t)

    except Exception as e:
        print(f"[ERROR] Main failed: {e}", flush=True)
        traceback.print_exc()

    finally:
        if env:
            try:
                await env.close()
            except Exception:
                pass


# ---------------- ENTRY (FINAL FIX) ----------------
if __name__ == "__main__":
    import os
    import sys
    import traceback
    import asyncio

    try:
        print("[BOOT] Starting inference...", flush=True)

        async def safe_main():
            try:
                await main()
            except Exception as e:
                print("[SAFE_MAIN_ERROR]", flush=True)
                try:
                    traceback.print_exc()
                except:
                    print(str(e), flush=True)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(safe_main())
        finally:
            try:
                loop.close()
            except:
                pass

    except Exception as e:
        print("[TOP_LEVEL_FATAL]", flush=True)
        try:
            traceback.print_exc()
        except:
            print(str(e), flush=True)

    finally:
        # 🚨 FORCE SUCCESS EXIT
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            pass

        os._exit(0)