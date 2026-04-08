from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import CloudEnvAction, CloudEnvObservation, CloudEnvState
from typing import Dict, Any


class CloudEnvClient(EnvClient[CloudEnvAction, CloudEnvObservation, CloudEnvState]):

    # ---------------- STEP PAYLOAD ----------------
    def _step_payload(self, action: CloudEnvAction) -> dict:
        return {
            "command": action.command,
            "args": action.args
        }

    # ---------------- SAFE PARSE RESULT ----------------
    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation") or {}

        if not isinstance(obs_data, dict):
            obs_data = {}

        # -------- SAFE HELPERS --------
        def safe_str(val, default=""):
            if val is None:
                return default
            return str(val)

        def safe_float(val, default=0.0):
            try:
                return float(val)
            except:
                return default

        def safe_bool(val, default=False):
            return bool(val) if val is not None else default

        # -------- SAFE EXTRACTION --------
        error_msg = safe_str(obs_data.get("error") or payload.get("error"))

        reward = safe_float(payload.get("reward") or obs_data.get("reward"))
        done = safe_bool(payload.get("done") or obs_data.get("done"))

        return StepResult(
            observation=CloudEnvObservation(
                done=done,
                reward=reward,
                output=safe_str(obs_data.get("output")),
                error=error_msg,
                current_task=safe_str(obs_data.get("current_task"), "unknown"),
                message=safe_str(obs_data.get("message"))
            ),
            reward=reward,
            done=done,
        )

    # ---------------- SAFE PARSE STATE ----------------
    def _parse_state(self, payload: dict) -> CloudEnvState:

        def safe_list(val):
            return val if isinstance(val, list) else []

        def safe_dict(val):
            return val if isinstance(val, dict) else {}

        return CloudEnvState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
            current_task=str(payload.get("current_task", "unknown")),
            max_steps=int(payload.get("max_steps", 30)),
            volumes=safe_list(payload.get("volumes")),
            security_groups=safe_list(payload.get("security_groups")),
            instances=safe_list(payload.get("instances")),
            logs=safe_dict(payload.get("logs"))
        )