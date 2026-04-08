from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import CloudEnvAction, CloudEnvObservation, CloudEnvState
from typing import Dict, Any

class CloudEnvClient(EnvClient[CloudEnvAction, CloudEnvObservation, CloudEnvState]):
    def _step_payload(self, action: CloudEnvAction) -> dict:
        return {
            "command": action.command,
            "args": action.args
        }

    def _parse_result(self, payload: dict) -> StepResult:
        # The server might return the observation nested or flat.
        obs_data = payload.get("observation", {})
        if not isinstance(obs_data, dict):
            obs_data = {}

        # Capture fields from either local obs_data or top-level payload
        def get_field(key, default=""):
            return str(obs_data.get(key) or payload.get(key) or default)

        error_msg = get_field("error")
        # Ensure we favor the top-level reward/done if present
        reward = float(payload.get("reward") or obs_data.get("reward") or 0.0)
        done = bool(payload.get("done") or obs_data.get("done") or False)
            
        return StepResult(
            observation=CloudEnvObservation(
                done=done,
                reward=reward,
                output=get_field("output"),
                error=error_msg,
                current_task=get_field("current_task", "unknown"),
                message=get_field("message")
            ),
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> CloudEnvState:
        return CloudEnvState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task=payload.get("current_task", "unknown"),
            max_steps=payload.get("max_steps", 30),
            volumes=payload.get("volumes", []),
            security_groups=payload.get("security_groups", []),
            instances=payload.get("instances", []),
            logs=payload.get("logs", {})
        )
