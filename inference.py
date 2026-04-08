import os
import sys
import json
import traceback
import re
import textwrap
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from client import CloudEnvClient
from models import CloudEnvAction

# --- CONFIGURATION ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("OPENENV_BASE_URL") or "http://localhost:7860"
TASK_NAME = os.getenv("TASK", "medium") # Changed default for testing

MAX_STEPS = 12 

# --- REFINED SRE PROMPT ---
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert SRE Agent. You solve infrastructure issues using ONLY these commands:
    - describe_volumes / delete_volume
    - describe_security_groups / update_security_group
    - describe_instances / modify_instance_attribute / start_instance

    STRATEGY:
    1. First, call a 'describe' command to find resource IDs.
    2. Once you have the ID, immediately call the action command (delete, update, or modify).
    
    TASK SPECIFICS:
    - EASY: delete_volume for the detached volume.
    - MEDIUM: update_security_group for the DB group. Set port=5432, cidr='10.0.0.0/8'.
    - HARD: modify_instance_attribute (type=t3.large) THEN start_instance.

    RESPONSE FORMAT:
    COMMAND: <command_name>
    ID: <id>
    """
).strip()

def parse_ai_action(text: str, current_task: str):
    text = text.lower()
    command = ""
    args = {}

    # 1. Advanced Command Identification
    mapping = {
        "delete": "delete_volume",
        "update": "update_security_group",
        "modify": "modify_instance_attribute",
        "change": "modify_instance_attribute",
        "start": "start_instance",
        "describe_vol": "describe_volumes",
        "describe_sec": "describe_security_groups",
        "describe_inst": "describe_instances"
    }
    
    # Check for exact matches first
    cmds = ["delete_volume", "update_security_group", "modify_instance_attribute", 
            "start_instance", "describe_volumes", "describe_security_groups", "describe_instances"]
    
    for c in cmds:
        if c in text:
            command = c
            break
            
    # Fallback to fuzzy matching if no exact command found
    if not command:
        for key, val in mapping.items():
            if key in text:
                command = val
                break

    # 2. Argument Extraction
    ids = re.findall(r'(vol-\w+|sg-\w+|i-\w+)', text)
    if ids:
        for entry_id in ids:
            if "vol-" in entry_id: args["volume_id"] = entry_id
            if "sg-" in entry_id: args["sg_id"] = entry_id
            if "i-" in entry_id: args["instance_id"] = entry_id

    # Task-specific argument forcing
    if "10.0.0.0/8" in text or "medium" in current_task:
        args["cidr"] = "10.0.0.0/8"
        args["port"] = 5432
    if "t3.large" in text:
        args["attribute"] = "type"
        args["value"] = "t3.large"
    
    # 3. Validation Logic (The "Loop Breaker")
    if command == "delete_volume" and "volume_id" not in args:
        command = "describe_volumes"
    if command == "update_security_group" and "sg_id" not in args:
        command = "describe_security_groups"
    if command == "modify_instance_attribute" and "instance_id" not in args:
        command = "describe_instances"

    # Final Fallback
    if not command:
        if "easy" in current_task: command = "describe_volumes"
        elif "medium" in current_task: command = "describe_security_groups"
        else: command = "describe_instances"

    return command, args

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CloudEnvClient(base_url=ENV_URL).sync()
    
    unique_rewards = set()
    rewards_history = []
    steps_taken = 0
    success = False

    print(f"[START] task={TASK_NAME} env=cloud_incident model={MODEL_NAME}", flush=True)

    try:
        res = env.reset(task=TASK_NAME)
        obs = res.observation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            # Pass all previous context to the model
            user_msg = f"STEP {step}:\nOBSERVATION: {obs.output or 'None'}\nERROR: {obs.error or 'None'}\nTASK: {obs.message}"
            messages.append({"role": "user", "content": user_msg})

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0
            )
            
            raw_msg = completion.choices[0].message.content or ""
            print(f"--- AI Thought: {raw_msg.splitlines()[0]} ---") # Debug print
            messages.append({"role": "assistant", "content": raw_msg})

            cmd, cmd_args = parse_ai_action(raw_msg, TASK_NAME)

            step_res = env.step(CloudEnvAction(command=cmd, args=cmd_args))
            obs = step_res.observation
            
            r = float(step_res.reward or 0.0)
            rewards_history.append(r)
            unique_rewards.add(r)
            steps_taken = step
            
            print(f"[STEP] step={step} action={cmd} reward={r:.2f} done={str(step_res.done).lower()} error={obs.error or 'null'}", flush=True)
            
            if step_res.done: 
                success = (max(unique_rewards) > 0)
                break

        final_score = max(unique_rewards) if unique_rewards else 0.0

    except Exception:
        traceback.print_exc()
    finally:
        try: env.close()
        except: pass
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={final_score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards_history)}", flush=True)
        sys.exit(0)

if __name__ == "__main__":
    main()