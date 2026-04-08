from typing import List, Dict, Optional, Any
from pydantic import Field, ConfigDict
from openenv import Action, Observation, State

class CloudEnvAction(Action):
    model_config = ConfigDict(protected_namespaces=())
    """
    An action executed by the simulated SRE agent.
    command: The name of the CLI/API command to execute.
    args: Key-value pairs of arguments to pass to the command.
    """
    command: str
    args: Dict[str, Any] = Field(default_factory=dict)

class CloudEnvObservation(Observation):
    model_config = ConfigDict(protected_namespaces=())
    """
    The observation returned by the cloud simulator after an action.
    """
    # Note: done and reward are inherited from Observation
    output: str           # Simulated standard output/JSON result
    error: str = ""       # Execution error if any
    current_task: str     # The difficulty level of the active task
    message: str          # Feedback message (e.g., 'Task completed successfully!')

class CloudEnvState(State):
    model_config = ConfigDict(protected_namespaces=())
    """
    Internal state of the simulated cloud infrastructure.
    """
    # Note: episode_id and step_count are inherited from State
    current_task: str = "easy"
    max_steps: int = 30
    
    # Simulated resource schemas
    volumes: List[Dict[str, Any]] = Field(default_factory=list)
    security_groups: List[Dict[str, Any]] = Field(default_factory=list)
    instances: List[Dict[str, Any]] = Field(default_factory=list)
    logs: Dict[str, str] = Field(default_factory=dict) # instance_id -> logs