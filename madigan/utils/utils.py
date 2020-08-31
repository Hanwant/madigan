import json
from pathlib import Path
import numpy as np
import torch


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_json(path):
    with open(path, 'r') as f:
        out = json.load(f)
    return out

def save_json(obj, path, write_mode='w'):
    with open(path, write_mode) as f:
        json.dump(obj, f)

def batchify_sarsd(sarsd):
    sarsd.state.price = sarsd.state.price[None, ...]
    sarsd.state.port = sarsd.state.port[None, ...]
    sarsd.action = sarsd.action[None, ...]
    sarsd.reward = np.array(sarsd.reward)[None, ...]
    sarsd.next_state.price = sarsd.next_state.price[None, ...]
    sarsd.next_state.port = sarsd.next_state.port[None, ...]
    sarsd.done = np.array(sarsd.done)[None, ...]
    return sarsd
