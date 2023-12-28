import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

if __name__ == '__main__':
    # 无缺失数据,异常值暂不测试
    raw_data = pd.read_csv('../Data/train.csv', header=0)
    print(raw_data.isnull().sum())
    word_dic = {}
    datas = list(raw_data['text'])
    al_len = 0
    lent = -1
    for data in datas:
        tmp = data.split()
        al_len += len(tmp)
        if len(tmp) > lent:
            lent = len(tmp)
    print(al_len / len(datas))
    print(lent)
