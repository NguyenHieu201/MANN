import torch
import torch.nn as nn

from Module.controller import Controller
from Module.head import Head
from Module.memory import Memory


"""
    Memory update with sample
"""

class MANN(nn.Module):
    def __init__(self, input_dim, output_dim, ctrl_dim, locations, location_size, gamma):
        super().__init__()
        
        self.controller = Controller(input_dim, ctrl_dim)
        
        self.head = Head(ctrl_dim, location_size, gamma)
        
        self.memory = Memory(locations, location_size)
        
        self.out_net = nn.Linear(ctrl_dim + location_size, output_dim)
        
        self.prev_read = torch.zeros(size=(1, locations))
        self.prev_write = torch.zeros(size=(1, locations))
        self.prev_lu = torch.zeros(size=(1, locations))
        self.prev_usage = torch.zeros(size=(1, locations))
        
        self.gate = 0
        self.gamma = gamma
        
    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.zeros((batch_size, ))
        for i in range(batch_size):
            input = x[i]
            h, gate = self.controller(input)
            
            # Read head
            read_key = self.head(h)
            w_read, r = self.memory.query_read_weight(read_key)
            
            r = r.squeeze()
            h = h.view(-1)
            r = r.view(-1)
            
            read_data = torch.cat([h, r], dim=0)
            output[i] = self.out_net(read_data)
        return output, h, gate, w_read
        
        
    def update_memory(self, h, gate, w_read):
        # Write head
            write_key = self.head.write_net(h)
            gate = torch.sigmoid(gate)
            w_write = gate * self.prev_read + (1 - gate) * self.prev_lu
            self.memory.update_memory(write_key, w_write)
            
            self.prev_read = w_read
            self.prev_write = w_write
            self.prev_usage = self.gamma * self.prev_usage + w_read + w_write
            self.prev_lu = self.compute_least_use(self.prev_lu)
    
    @staticmethod
    def compute_least_use(w_u):
        min_u = torch.min(w_u)
        lu_mask = torch.where(w_u <= min_u)
        non_lu_mask = torch.where(w_u > min_u)
        w_u[lu_mask] = 1
        w_u[non_lu_mask] = 0
        w_lu = w_u
        return w_lu