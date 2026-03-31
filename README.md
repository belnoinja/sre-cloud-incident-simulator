---
title: Cloud Incident Simulator
emoji: 🚨
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Cloud Infrastructure Incident Simulator 🌩️🚨

A real-world OpenEnv simulator challenging AI agents to act as Junior SREs (Site Reliability Engineers). The agent must navigate a simulated cloud environment via an API, analyze issues, and run corrective commands to resolve incidents—without breaking production.

## Motivation
Most RL/agent environments are games or toy text tasks. This environment tests realistic, multi-step troubleshooting logic, requiring the agent to:
1. Examine current infrastructure state (read).
2. Correlate logs with instance configurations (analyze).
3. Safely apply patches or architectural updates (act).

## Space Definitions
Strictly typed via Pydantic models.

### Action Space (`CloudEnvAction`)
Agents pass a discrete command and keyword arguments.
- `command` (str): One of `describe_volumes`, `delete_volume`, `describe_security_groups`, `update_security_group`, `describe_instances`, `read_logs`, `modify_instance_attribute`, `start_instance`.
- `args` (dict): Keyword configurations (e.g. `{"volume_id": "vol-123"}`).

### Observation Space (`CloudEnvObservation`)
- `output` (str): Command results (often JSON strings representing Cloud properties).
- `error` (str): Error message if the action failed.
- `current_task` (str): The active difficulty level.
- `message` (str): The environment feedback or initialize objective.
- `reward` (float): Standard OpenEnv reward [0.0 - 1.0], -1.0 for destructive fails.
- `done` (bool): True if completed, failed, or timed out.

## Tasks
The environment features 3 distinct tasks to test varying levels of reasoning depth:

1. **Easy (Cost Optimization)**: 
   - **Scenario**: The cloud account is cluttered with orphaned EBS volumes costing money.
   - **Goal**: Find and delete all unattached volumes without touching in-use storage.
2. **Medium (Security Incident)**: 
   - **Scenario**: A production database is accidentally exposed to `0.0.0.0/0`.
   - **Goal**: Identify the database security group and restrict inbound access to `10.0.0.0/8` without modifying the frontend web groups.
3. **Hard (Outage Resolution)**: 
   - **Scenario**: A critical worker node crashed and won't turn on.
   - **Goal**: Read the logs to identify an OOM error, resize the instance's type to provision more RAM, and restart the instance.

## Setup Instructions

1. **Prerequisites**: Python 3.10+ and an active Hugging Face account (for deployment).
2. **Install**:
   ```bash
   git clone <repo-url>
   cd cloud_env
   pip install -r requirements.txt
   ```

## Usage

### Local Development Server
Boot up the simulated backend using Uvicorn or UV:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
# or if using openenv framework directly natively:
# uv run server
```

### Baseline Inference Test
Run the OpenAI implementation evaluating the agent across all 3 tasks (ensure the server from above is running in another terminal).
```bash
export OPENAI_API_KEY="sk-..."
export API_BASE_URL="http://localhost:8000"
python inference.py
```

## Baseline Scores (GPT-4o)
- **Easy**: 1.0 (Pass)
- **Medium**: 1.0 (Pass) 
- **Hard**: 1.0 (Pass)
*(Note: Less capable models often fail the Hard task because they try starting the instance repeatedly without reading the logs to discover the OOM error.)*
