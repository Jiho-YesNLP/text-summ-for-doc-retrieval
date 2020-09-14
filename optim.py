"""Optimizer class that can contain multiple optimizers for different models"""
from transformers import AdamW

class Optimizers:
    def __init__(self):
        self.optims = []

    def add_optimizer(self, params, method='adamw', lr=1e-3):
        if method == 'adamw':
            self.optims.append(AdamW(params, lr=lr))
        else:
            raise NotImplementedError
