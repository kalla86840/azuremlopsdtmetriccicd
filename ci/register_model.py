from azureml.core import Workspace, Model
import json

def load_config(path='config.json'):
    import os
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, path)) as f:
        return json.load(f)

def register_model():
    config = load_config()
    ws = Workspace.get(
        name=config['workspace_name'],
        subscription_id=config['subscription_id'],
        resource_group=config['resource_group']
    )
    model = Model.register(
        workspace=ws,
        model_path='model.joblib',
        model_name='decision_tree_model'
    )
    print(f"Model registered: {model.name}, version: {model.version}")

if __name__ == '__main__':
    register_model()
