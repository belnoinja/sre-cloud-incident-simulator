from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import CloudEnvAction, CloudEnvObservation, CloudEnvState

class CloudEnvClient(EnvClient[CloudEnvAction, CloudEnvObservation, CloudEnvState]):
    
    # ---------------- STEP PAYLOAD ----------------
    def _step_payload(self, action: CloudEnvAction) -> dict:
        """
        Returns only the command and args. 
        The unified openenv library handles wrapping this in the 'action' key 
        automatically for HTTP/MCP transport.
        """
        return {
            "command": action.command,
            "args": action.args
        }

    # ---------------- SAFE PARSE RESULT ----------------
    def _parse_result(self, payload: dict) -> StepResult:
        """
        Parses the JSON response from the server back into a StepResult object.
        Optimized for openenv>=0.1.13.
        """
        obs_data = payload.get("observation", {})
        
        # Ensure we don't crash if observation comes back as something other than a dict
        if not isinstance(obs_data, dict):
            obs_data = {}

        # Extract metrics safely
        reward = float(payload.get("reward", 0.0) or 0.0)
        # Check both the top-level 'done' and the nested observation 'done'
        done = bool(payload.get("done", False) or obs_data.get("done", False))

        return StepResult(
            observation=CloudEnvObservation(
                done=done,
                reward=reward,
                output=str(obs_data.get("output", "")),
                error=str(obs_data.get("error", "")),
                current_task=str(obs_data.get("current_task", "unknown")),
                message=str(obs_data.get("message", ""))
            ),
            reward=reward,
            done=done,
        )

    # ---------------- SAFE PARSE STATE ----------------
    def _parse_state(self, payload: dict) -> CloudEnvState:
        """
        Maps the raw state payload directly to your CloudEnvState Pydantic model.
        """
        return CloudEnvState(**payload)