from Module.mann import MANN

from Utils import *
from Loader import get_set_and_loader
import json
import torch
from torch import optim
from torch.nn import MSELoss
from tqdm import tqdm
import matplotlib.pyplot as plt

config_path = "./config.json"
config = json.load(open(config_path, 'r'))

data = get_data(folder="./Data/stock_data/data_for_teacher",
                src_name="ACB")

data = preprocessing(data=data, name="ACB", test_ratio=0.3,
                     mode="one-day", setting=config)

dataset, dataloader = get_set_and_loader(X=data["train"]["X"],
                                         Y=data["train"]["Y"],
                                         batch_size=1)

input_dim = config["input-dim"]
ctrl_dim = config["ctrl-dim"]
output_dim = config["output-dim"]
read_data_size = config["read-data-size"]
locations = config["locations"]
location_size = config["location-size"]
gamma = config["gamma"]

save_path = "./Weight/mann.pth"



mann = MANN(input_dim, output_dim, ctrl_dim, locations, location_size, gamma)

mann.load_state_dict(torch.load(save_path))
mann.eval()
input = torch.tensor(data["test"]["X"])
prediction, h, gate, w_read = mann(input)
prediction = prediction.detach().numpy()
true_value = data["test"]["Y"]
plt.plot(true_value, 'k', label="True value")
plt.plot(prediction, 'r', label="Prediction")
plt.legend()
plt.show()


# loss_fn = MSELoss()
# optimizer = optim.Adam(mann.parameters())

# epochs = 10

# for epoch in tqdm(range(epochs)):
#     for x, y in dataloader:
#         optimizer.zero_grad()
#         output, h, gate, w_read = mann(x)
#         loss = loss_fn(output, y.view(-1))
#         loss.backward()
#         optimizer.step()
#         mann.update_memory(h, gate, w_read)
        
        
# save model
# torch.save(mann.state_dict(), save_path)

# input = torch.tensor(data["test"]["X"])
# prediction, h, gate, w_read = mann(input)
# prediction = prediction.detach().numpy()
# true_value = data["test"]["Y"]
# plt.plot(true_value, 'k', label="True value")
# plt.plot(prediction, 'r', label="Prediction")
# plt.legend()
# plt.show()
    