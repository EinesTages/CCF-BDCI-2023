import numpy as np
import pandas as pd

a = pd.read_csv("./Data/children.csv")
dic = {}
wait = []
for i, row in a.iterrows():
    if not row['label'].isnull():
        dic[row['node_id']] = row['label']
    else:
        wait.append(row['node_id'])
while True:
    improve = False
    for i in range(0,len(wait)):
        j = eval(a[i]['neighbour'])
