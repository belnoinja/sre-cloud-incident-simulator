import os
import sys
import uvicorn
from openenv.core.env_server import create_fastapi_app


# Pathing fix: Ensures that 'models.py' and 'server/' are discoverable 
# regardless of where the validator initiates the process.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import CloudIncidentEnvironment
from models import CloudEnvAction, CloudEnvObservation

# Initialize the application with the unified server helper
app = create_fastapi_app(CloudIncidentEnvironment, CloudEnvAction, CloudEnvObservation)

def main():
    # Hugging Face and modern cloud environments default to 7860
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()