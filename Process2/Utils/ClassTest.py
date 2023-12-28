import pandas as pd

if __name__ == '__main__':
    raw_data = pd.read_csv('../Data/train.csv', header=0)
    node_dic = {}
    datas = raw_data['label']
    for data in datas:
        if data not in node_dic:
            node_dic[data] = 1
        else:
            node_dic[data] = node_dic[data] + 1
    print(node_dic)
