import numpy as np

a = np.load("./Graph_Data/feature.npy")
b = []
for item in a:
    b.append(item[0].astype("float32"))
np.save("./Graph_Data/feature6.npy",np.array(b))