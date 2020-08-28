import json
from pathlib import Path


def load_json(path):
    with open(path, 'r') as f:
        out = json.load(f)
    return out

def save_json(obj, path, write_mode='w'):
    with open(path, write_mode) as f:
        json.dump(obj, f)
