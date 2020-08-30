import json
from pathlib import Path
import numpy as np
from ..fleet.conv_model import ConvModel
from ..fleet.mlp_model import MLPModel



def load_json(path):
    with open(path, 'r') as f:
        out = json.load(f)
    return out

def save_json(obj, path, write_mode='w'):
    with open(path, write_mode) as f:
        json.dump(obj, f)

def get_model_class(name):
    if name == "ConvModel":
        return ConvModel
    elif name == "MLPModel":
        return MLPModel
    else:
        raise NotImplementedError(f"model {name} is not Implemented")
