import joblib
import numpy as np
import json
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('decision_tree_model')  # match registered model name
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        predictions = model.predict(data)
        return predictions.tolist()
    except Exception as e:
        return str(e)
