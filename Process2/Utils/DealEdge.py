import numpy as np
import pandas as pd

def Edge_Out(Edges):
    edge_x = []
    edge_y = []
    i = 0
    for item in Edges:
        ans = eval(item)
        for j in ans:
            edge_x.append(i)
            edge_y.append(j)
        i = i + 1
    return edge_x, edge_y


if __name__ == '__main__':
    raw_data1 = pd.read_csv('../Data/children.csv', header=0)
    raw_data2 = pd.read_csv('../Data/train.csv',header=0)
    raw_data3 = pd.read_csv('../Data/test.csv', header=0)
    x, y = Edge_Out(raw_data1['neighbour'])
    df1 = pd.DataFrame()
    df1.loc[:, 0] = x
    df1.loc[:, 1] = y
    df1.to_csv('../Graph_Data/edges.csv', index=False)
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df2['nid'] = raw_data2['node_id']
    df2['label'] = raw_data2['label']
    df3['nid'] = raw_data3['node_id']
    df2.to_csv('../Graph_Data/tr.csv', index=False)
    df3.to_csv('../Graph_Data/te.csv', index=False)