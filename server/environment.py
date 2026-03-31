import json
import uuid
import copy
from typing import Dict, Any, Tuple
from openenv.core.env_server import Environment
from models import CloudEnvAction, CloudEnvObservation, CloudEnvState

class CloudIncidentEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    MAX_STEPS = 30

    def __init__(self):
        self._state = CloudEnvState()
        self._won = False
        self._lost = False

    def reset(self, seed=None, episode_id=None, **kwargs) -> CloudEnvObservation:
        task = kwargs.get("task", "easy").lower()
        self._state = CloudEnvState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task=task,
            max_steps=self.MAX_STEPS,
            volumes=[],
            security_groups=[],
            instances=[],
            logs={}
        )
        self._won = False
        self._lost = False

        message = ""
        # Initialize resources based on task
        if task == "easy":
            self._state.volumes = [
                {"id": "vol-001", "size": 100, "state": "in-use", "instance_id": "i-abc"},
                {"id": "vol-002", "size": 500, "state": "available", "instance_id": None}, # Target 1
                {"id": "vol-003", "size": 50, "state": "in-use", "instance_id": "i-xyz"},
                {"id": "vol-004", "size": 1000, "state": "available", "instance_id": None}, # Target 2
            ]
            message = "Task [Easy]: Find and delete all unattached (state='available') storage volumes to save costs."
        
        elif task == "medium":
            self._state.security_groups = [
                {
                    "id": "sg-db01",
                    "name": "production-database-sg",
                    "inbound_rules": [
                        {"port": 5432, "protocol": "tcp", "cidr": "0.0.0.0/0"} # Vulnerability
                    ]
                },
                {
                    "id": "sg-web01",
                    "name": "frontend-web-sg",
                    "inbound_rules": [
                        {"port": 80, "protocol": "tcp", "cidr": "0.0.0.0/0"},
                        {"port": 443, "protocol": "tcp", "cidr": "0.0.0.0/0"}
                    ]
                }
            ]
            message = "Task [Medium]: The 'production-database-sg' security group is publicly exposed. Restrict port 5432 to internal subnet '10.0.0.0/8'. Do not touch frontend-web-sg."
            
        elif task == "hard":
            self._state.instances = [
                {"id": "i-worker01", "type": "t2.micro", "state": "stopped", "memory": "1GB"}
            ]
            self._state.logs["i-worker01"] = "[ERROR] OutOfMemoryError. The process crashed. Instance lacks sufficient RAM."
            message = "Task [Hard]: The worker instance 'i-worker01' crashed due to OOM. Read its logs, upgrade its instance type to 't3.large' (8GB RAM), and start it."
        
        else:
            message = f"Unknown task: {task}. Valid tasks are: easy, medium, hard."
            self._state.current_task = "invalid"

        return CloudEnvObservation(
            done=False,
            reward=None,
            output="Environment initialized.",
            error="",
            current_task=self._state.current_task,
            message=message
        )

    def step(self, action: CloudEnvAction, timeout_s=None, **kwargs) -> CloudEnvObservation:
        self._state.step_count += 1
        
        # Dispatch command
        output_str = ""
        error_str = ""
        
        try:
            cmd = action.command.lower()
            if cmd == "describe_volumes":
                output_str = json.dumps(self._state.volumes, indent=2)
            elif cmd == "delete_volume":
                vid = action.args.get("volume_id")
                idx = next((i for i, v in enumerate(self._state.volumes) if v["id"] == vid), -1)
                if idx >= 0:
                    if self._state.volumes[idx]["state"] == "in-use":
                        error_str = f"Volume {vid} is in-use!"
                        self._lost = True # Destructive
                    else:
                        self._state.volumes.pop(idx)
                        output_str = f"Volume {vid} deleted successfully."
                else:
                    error_str = f"Volume {vid} not found."
            
            elif cmd == "describe_security_groups":
                output_str = json.dumps(self._state.security_groups, indent=2)
            
            elif cmd == "update_security_group":
                sg_id = action.args.get("sg_id")
                port = action.args.get("port")
                new_cidr = action.args.get("cidr")
                idx = next((i for i, sg in enumerate(self._state.security_groups) if sg["id"] == sg_id), -1)
                if idx >= 0:
                    for rule in self._state.security_groups[idx]["inbound_rules"]:
                        if rule["port"] == port:
                            rule["cidr"] = new_cidr
                    output_str = f"Updated rule on {sg_id} port {port} to {new_cidr}"
                else:
                    error_str = f"Security Group {sg_id} not found."
            
            elif cmd == "describe_instances":
                output_str = json.dumps(self._state.instances, indent=2)
                
            elif cmd == "read_logs":
                iid = action.args.get("instance_id")
                if iid in self._state.logs:
                    output_str = self._state.logs[iid]
                else:
                    error_str = f"Logs for {iid} not found."
            
            elif cmd == "modify_instance_attribute":
                iid = action.args.get("instance_id")
                attr = action.args.get("attribute") # e.g. "type"
                val = action.args.get("value")      # e.g. "t3.large"
                idx = next((i for i, inst in enumerate(self._state.instances) if inst["id"] == iid), -1)
                if idx >= 0:
                    self._state.instances[idx][attr] = val
                    if attr == "type" and val == "t3.large":
                        self._state.instances[idx]["memory"] = "8GB"
                    output_str = f"Modified {attr} of {iid} to {val}."
                else:
                    error_str = f"Instance {iid} not found."
                    
            elif cmd == "start_instance":
                iid = action.args.get("instance_id")
                idx = next((i for i, inst in enumerate(self._state.instances) if inst["id"] == iid), -1)
                if idx >= 0:
                    if self._state.instances[idx]["memory"] == "1GB":
                        error_str = "Status Check Failed: Instance immediately crashed again (OOM). Need more memory!"
                        self._state.instances[idx]["state"] = "stopped"
                    else:
                        self._state.instances[idx]["state"] = "running"
                        output_str = f"Instance {iid} started successfully."
                else:
                    error_str = f"Instance {iid} not found."
            else:
                error_str = f"Unknown command: {cmd}. Available: describe_volumes, delete_volume, describe_security_groups, update_security_group, describe_instances, read_logs, modify_instance_attribute, start_instance"
        except Exception as e:
            error_str = f"Exception executing command: {str(e)}"
            
        # Grade the state
        done, reward, msg = self._grade()
        
        if self._state.step_count >= self.MAX_STEPS and not done:
            done = True
            reward = reward or 0.0
            msg = "Max steps reached without success."

        return CloudEnvObservation(
            done=done,
            reward=reward,
            output=output_str,
            error=error_str,
            current_task=self._state.current_task,
            message=msg
        )

    def _grade(self) -> Tuple[bool, float, str]:
        if self._lost:
            return True, -1.0, "Critical failure! You performed a destructive action on a production resource."
            
        task = self._state.current_task
        
        if task == "easy":
            # Target: all available volumes are deleted, and no in-use volumes are harmed.
            available_remaining = sum(1 for v in self._state.volumes if v["state"] == "available")
            in_use_remaining = sum(1 for v in self._state.volumes if v["state"] == "in-use")
            if in_use_remaining < 2:
                self._lost = True
                return True, -1.0, "You deleted an in-use volume! Production outage."
                
            if available_remaining == 0:
                self._won = True
                return True, 1.0, "Success! All unattached volumes deleted safely."
            else:
                # Partial reward: proportion of deleted available volumes
                initial_available = 2
                deleted = initial_available - available_remaining
                return False, 0.5 * (deleted / initial_available), "Working on it..."
                
        elif task == "medium":
            sg_db = next((sg for sg in self._state.security_groups if sg["id"] == "sg-db01"), None)
            if sg_db:
                rule = sg_db["inbound_rules"][0]
                if rule["cidr"] == "10.0.0.0/8" and rule["port"] == 5432:
                    self._won = True
                    return True, 1.0, "Success! The database security group is restricted."
                elif rule["cidr"] != "0.0.0.0/0" and rule["cidr"] != "10.0.0.0/8":
                    # They changed it but to the wrong CIDR
                    return False, 0.5, "Partially restricted, but to the wrong CIDR."
            return False, 0.0, "Working on it..."
            
        elif task == "hard":
            inst = next((i for i in self._state.instances if i["id"] == "i-worker01"), None)
            if inst:
                if inst["state"] == "running" and inst["type"] == "t3.large":
                    self._won = True
                    return True, 1.0, "Success! Instance upgraded and safely running."
                elif inst["type"] == "t3.large":
                    return False, 0.5, "Instance upgraded, but not yet started."
            return False, 0.0, "Working on it..."
            
        return True, 0.0, "Invalid task."

    @property
    def state(self) -> CloudEnvState:
        return self._state
