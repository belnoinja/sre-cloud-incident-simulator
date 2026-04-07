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
        obs_data = payload.get("observation", {})
        
        # In case the result directly holds the error message (as a failed step response)
        error_msg = obs_data.get("error", "")
        # fallback if format changes
        if not obs_data and "error" in payload:
             error_msg = payload["error"]

        if not isinstance(obs_data, dict):
            obs_data = {}
            
        return StepResult(
            observation=CloudEnvObservation(
                done=payload.get("done", False),
                reward=float(payload.get("reward") or 0.0),
                output=str(obs_data.get("output", "")),
                error=error_msg,
                current_task=str(obs_data.get("current_task") or "unknown"),
                message=str(obs_data.get("message", ""))
            ),
            reward=float(payload.get("reward") or 0.0),
            done=payload.get("done", False),
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
