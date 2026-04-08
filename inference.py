import asyncio
import json
import os
import sys
import traceback
from typing import List

from openai import AsyncOpenAI
from client import CloudEnvClient
from models import CloudEnvAction

# 🔥 BOOT LOG (prevents "unknown" status)
print("[BOOT] script loaded", flush=True)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

# 🔥 YOUR HF SPACE
ENV_URL = "https://belnoinja-cloud-incident-simulator.hf.space"

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


# 🔥 WAIT FOR HF SPACE (handles cold start)
async def wait_for_env(env, retries=15):
    for i in range(retries):
        try:
            res = await env.reset(task="easy")
            if res and res.observation:
                print("[DEBUG] env ready", flush=True)
                return True
        except Exception:
            print(f"[DEBUG] waiting for env... {i+1}", flush=True)
            await asyncio.sleep(2)
    return False


async def run_episode(client, env, task_name):
    print(f"[START] task={task_name}", flush=True)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        res = await env.reset(task=task_name)
        obs = res.observation

        messages = [
            {"role": "system", "content": "You are an SRE agent. Use tools."},
            {"role": "user", "content": obs.message}
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
                print(f"[STEP] step={step} action=llm_error reward=0.00 done=true error={str(e)}", flush=True)
                break

            if not msg.tool_calls:
                print(f"[STEP] step={step} action=stop reward=0.00 done=true error=no_tool_call", flush=True)
                break

            messages.append(msg)

            for tool_call in msg.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                    command = args.get("command")
                    cmd_args = args.get("args", {})
                except Exception as e:
                    print(f"[STEP] step={step} action=parse_error reward=0.00 done=false error={str(e)}", flush=True)
                    continue

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
                        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={obs.error or 'null'}",
                        flush=True
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": obs.output or obs.error or ""
                    })

                    if done:
                        score = max(0.0, min(1.0, reward))
                        success = score > 0
                        break

                except Exception as e:
                    print(
                        f"[STEP] step={step} action={action_str} reward=0.00 done=true error={str(e)}",
                        flush=True
                    )
                    break

            if success:
                break

    except Exception as e:
        print(f"[STEP] step=0 action=episode_error reward=0.00 done=true error={str(e)}", flush=True)

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}",
            flush=True
        )


async def main():
    try:
        # 🔥 ENV CHECK
        if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
            print("[STEP] step=0 action=missing_env reward=0.00 done=true error=missing_env_vars", flush=True)
            return

        client = AsyncOpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )

        env = CloudEnvClient(base_url=ENV_URL)

        # 🔥 WAIT FOR SPACE
        ready = await wait_for_env(env)
        if not ready:
            print("[STEP] step=0 action=env_not_ready reward=0.00 done=true error=env_failed", flush=True)
            return

        tasks = ["easy", "medium", "hard"]

        for task in tasks:
            await run_episode(client, env, task)

        await env.close()

    except Exception as e:
        print(f"[STEP] step=0 action=main_error reward=0.00 done=true error={str(e)}", flush=True)


# 🔥 FINAL SAFE ENTRY POINT (CRITICAL)
if __name__ == "__main__":
    print("[BOOT] inference started", flush=True)

    try:
        asyncio.run(main())
    except Exception:
        print("[STEP] step=0 action=fatal reward=0.00 done=true error=unhandled_exception", flush=True)
        try:
            traceback.print_exc()
        except:
            pass
    finally:
        sys.exit(0)