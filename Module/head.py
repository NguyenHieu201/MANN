from turtle import forward
import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, ctrl_dim, location_size, gamma):
        super().__init__()
        self.read_net = nn.Linear(ctrl_dim, location_size)
        self.write_net = nn.Linear(ctrl_dim, location_size)
        
        self.gamma = gamma
    
    def forward(self, h):
        read_key = self.read_net(h)
        return read_key
    
    def compute_write_key(self, h):
        write_key = self.write_net(h)
        return write_key