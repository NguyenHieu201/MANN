from turtle import forward
import torch
import torch.nn as nn

"""
    
"""

class Controller(nn.Module):
    def __init__(self, input_dim, ctrl_dim):
        super().__init__()
        self.controller_net = nn.Linear(input_dim, ctrl_dim)        
        self.gate_net = nn.Linear(ctrl_dim, 1)
        
    def forward(self, x):
        h = self.controller_net(x)
        gate = self.gate_net(h)
        return h, gate