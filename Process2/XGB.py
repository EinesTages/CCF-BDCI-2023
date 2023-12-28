import os.path

import pandas as pd
import csv
import numpy as np
import multiprocessing as mp
from config import Config
from Utils.TextProcessor import clean_text
from Utils.TextProcessor import tokenizer
from model import XGB
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    train_data = pd.read_csv('Data/train.csv')
    test_data = pd.read_csv('Data/test.csv')
    train_text = list(pd.read_csv('Log/train_x.csv', header=None).iloc[:, 0])
    test_text = list(pd.read_csv('Log/test_x.csv', header=None).iloc[:, 0])
    for i in range(0, len(train_text)):
        train_text[i] = str(train_text[i])
    for i in range(0, len(test_text)):
        test_text[i] = str(test_text[i])
    all_text = train_text + test_text
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=2000
    )
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    X_train = train_word_features
    Y_train = list(train_data['label'])
    X_test = test_word_features
    param = {'learning_rate': 0.05,  # (xgb’s “eta”)
             'objective': 'multi:softmax',
             'n_jobs': 16,
             'n_estimators': 300,  # 树的个数
             'max_depth': 10,
             'gamma': 0.5,  # 惩罚项中叶子结点个数前的参数，Increasing this value will make model more conservative.
             'reg_alpha': 0,
             # L1 regularization term on weights.Increasing this value will make model more conservative.
             'reg_lambda': 2,
             # L2 regularization term on weights.Increasing this value will make model more conservative.
             'min_child_weight': 1,  # 叶子节点最小权重
             'subsample': 0.8,  # 随机选择80%样本建立决策树
             'random_state': 1,  # 随机数
             'num_class': 24,
             'device': 'cuda:1',
             'tree_method': 'approx'
             }
    final_model = XGB(X_train, Y_train)
    final_model.train(param)
    preds = final_model.model.predict(X_test)
    df = pd.DataFrame()
    df['node_id'] = list(test_data['node_id'])
    df['label'] = preds
    df.to_csv('./Result/100000/submission.csv', index=False)
