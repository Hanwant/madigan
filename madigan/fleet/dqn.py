from pathlib import Path
import torch
from .agent import Agent
from .conv_model import ConvModel, MLPModel

class DQN(Agent):
    def __init__(self, params):
        agent_params = params['agent_params']
        self.params = agent_params
        self.model_params = agent_params['model_params']
        self.savepath = agent_params['savepath']

        if Path(self.savepath).is_file():
            config = torch.load(self.savepath)
            self.model_class = config['model_class']
            self.n_layers = config['n_layers']
            self.d_model = config['d_model']
        else:
            self.model_class = self.model_params['model_class']
            self.n_layers = self.model_params['n_layers']
            self.d_model = self.model_params['d_model']

        if self.model_class == "ConvModel":
            self.model_b = ConvModel(d_model=self.d_model, n_layers=self.n_layers)
            self.model_t = ConvModel(d_model=self.d_model, n_layers=self.n_layers)
        elif self.model_class == "MLPModel":
            self.model_b = ConvModel(d_model=self.d_model, n_layers=self.n_layers)
            self.model_t = ConvModel(d_model=self.d_model, n_layers=self.n_layers)
        else:
            raise NotImplementedError(f"model_class: {self.model_class} is not Implemented")

        if Path(self.savepath).is_file():
            self.load_state()
        else:
            self.save_state()


    def save_state(self):
        config = {'state_dict': self.model_t.state_dict(), 'model_class': self.model_class,
                  'd_model': self.d_model, 'training_steps': self.training_steps,
                  'total_steps': self.total_steps}
        torch.save(config, self.savepath)

    def load_state(self):
        state = torch.load(self.savepath)
        self.model_b.load_state_dict(state['state_dict'])
        self.model_t.load_state_dict(state['state_dict'])
        self.training_steps = state['training_steps']
        self.total_steps = state['total_steps']

    def train_step(self, sarsd):
        pass
