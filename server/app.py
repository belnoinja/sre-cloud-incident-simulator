from openenv.core.env_server import create_fastapi_app
from server.environment import CloudIncidentEnvironment
from models import CloudEnvAction, CloudEnvObservation
import uvicorn
import os

app = create_fastapi_app(CloudIncidentEnvironment, CloudEnvAction, CloudEnvObservation)

def main():
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
