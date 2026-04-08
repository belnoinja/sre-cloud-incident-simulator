import asyncio
import json
import os
from typing import List

from openai import AsyncOpenAI
from client import CloudEnvClient
from models import CloudEnvAction

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
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


# 🔥 WAIT FOR ENV READY
async def wait_for_env(env, retries=10):
    for i in range(retries):
        try:
            await env.reset(task="easy")
            print("[DEBUG] Env ready", flush=True)
            return True
        except:
            print(f"[DEBUG] waiting for env... {i+1}", flush=True)
            await asyncio.sleep(1)
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
                print(f"[STEP] step={step} action=error reward=0.00 done=true error={str(e)}", flush=True)
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
        print(f"[STEP] step=0 action=fatal reward=0.00 done=true error={str(e)}", flush=True)

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}",
            flush=True
        )


async def main():
    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

    env_base = os.getenv("OPENENV_BASE_URL") or "http://localhost:7860"
    env = CloudEnvClient(base_url=env_base)

    # 🔥 WAIT FOR SERVER
    ready = await wait_for_env(env)
    if not ready:
        print("[STEP] step=0 action=env_not_ready reward=0.00 done=true error=env_failed", flush=True)
        return

    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        await run_episode(client, env, task)

    await env.close()


if __name__ == "__main__":
    asyncio.run(main())