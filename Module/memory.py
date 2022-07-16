import torch
import torch.nn as nn


class Memory(nn.Module):
    def __init__(self, locations, location_size):
        super().__init__()
        
        self.memory = torch.zeros([locations, location_size])
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        
        
    def query_read_weight(self, read_key):
        similarity = self.cos(self.memory, read_key)
        w_read = torch.exp(similarity)
        w_read = w_read / w_read.sum()
        w_read = torch.unsqueeze(w_read, dim=1)
        w_read = w_read.transpose(0, 1)
        r = torch.matmul(w_read, self.memory)
        return w_read, r
    
    def update_memory(self, write_key, write_weight):
        write_key = write_key.detach()
        write_weight = write_weight.detach()
        
        write_key = write_key.unsqueeze(0)
        self.memory = self.memory + torch.matmul(write_weight.transpose(0, 1), write_key)