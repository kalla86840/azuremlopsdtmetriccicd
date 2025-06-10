from azureml.core import Workspace
import json
import os

def load_config(path='config.json'):
    import os
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, path)) as f:
        return json.load(f)

def create_workspace():
    config = load_config()
    ws = Workspace.create(
        name=config['workspace_name'],
        subscription_id=config['subscription_id'],
        resource_group=config['resource_group'],
        location=config['location'],
        exist_ok=True,
        show_output=True
    )
    ws.write_config()
    print("Workspace configuration saved.")

if __name__ == '__main__':
    create_workspace()
