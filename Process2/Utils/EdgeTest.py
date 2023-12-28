import pandas as pd

if __name__ == '__main__':
    raw_data = pd.read_csv('../Data/train.csv', header=0)
    node_dic = {}
    other_dic = {}
    nodes = list(raw_data['node_id'])
    adjs = list(raw_data['neighbour'])
    for node in nodes:
        node_dic[node] = 1
    for adj in adjs:
        tmp = list(eval(adj))
        for other in tmp:
            if other not in node_dic:
                other_dic[other] = 1
    print(len(other_dic))
    # 这证明显然本题应该使用Children数据集,因为有12580个节点的信息都不在其中
