
from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
import json
import os


def load_config(path='config.json'):
    import os
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, '..', 'ci', path)) as f:
        return json.load(f)
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, '..', 'ci', path)) as f:
        return json.load(f)

def deploy_model():
    config = load_config()
    ws = Workspace.get(
        name=config['workspace_name'],
        subscription_id=config['subscription_id'],
        resource_group=config['resource_group']
    )
    model = Model(ws, name='decision_tree_model')

    env = Environment(name='deploy-env')
    env.python.conda_dependencies.add_pip_package('scikit-learn')
    env.python.conda_dependencies.add_pip_package('joblib')

    inference_config = InferenceConfig(entry_script='cd/score.py', environment=env)

    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
    service = Model.deploy(
        workspace=ws,
        name='dt-endpoint',
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        overwrite=True
    )
    service.wait_for_deployment(show_output=True)
    print(f"Service state: {service.state}")
    print(f"Scoring URI: {service.scoring_uri}")

if __name__ == '__main__':
    deploy_model()
