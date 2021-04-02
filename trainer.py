import torch

class Trainer():
    
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, x, y, config):
        self.model.train()

    def _evaluate(self, x, y, config):
        self.model.eval()

        