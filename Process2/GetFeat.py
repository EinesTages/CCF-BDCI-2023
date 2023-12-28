import torch
import numpy as np
a = torch.load("better.pt")
np.save("Graph_Data/feature10.npy",np.array(a))