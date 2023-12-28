from torch import nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
3

class XGB:

    def __init__(self, X_df, y_df):
        self.X = X_df
        self.y = y_df

    def train(self, param):
        self.model = XGBClassifier(**param)
        self.model.fit(self.X, self.y, eval_set=[(self.X, self.y)],
                       eval_metric=['mlogloss'],
                       early_stopping_rounds=10,  # 连续N次分值不再优化则提前停止
                       verbose=False
                       )

        #         mode evaluation
        train_result, train_proba = self.model.predict(self.X), self.model.predict_proba(self.X)
        train_acc = accuracy_score(self.y, train_result)
        train_auc = f1_score(self.y, train_result, average='macro')

        print("Train acc: %.2f%% Train auc: %.2f" % (train_acc * 100.0, train_auc))

    def test(self, X_test, y_test):
        result, proba = self.model.predict(X_test), self.model.predict_proba(X_test)
        acc = accuracy_score(y_test, result)
        f1 = f1_score(y_test, result, average='macro')

        print("acc: %.2f%% F1_score: %.2f%%" % (acc * 100.0, f1))

    def grid(self, param_grid):
        self.param_grid = param_grid
        xgb_model = XGBClassifier(nthread=20)
        clf = GridSearchCV(xgb_model, self.param_grid, scoring='f1_macro', cv=2, verbose=1)
        clf.fit(self.X, self.y)
        print("Best score: %f using parms: %s" % (clf.best_score_, clf.best_params_))
        return clf.best_params_, clf.best_score_


# 定义GCNConv网络,这里采用两层GCN
class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(num_node_features, 16)
        self.conv2 = pyg_nn.GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # [num_nodes, 16]
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)  # [num_nodes, num_classes]

        return F.log_softmax(x, dim=1)  # [num_nodes, num_classes]
