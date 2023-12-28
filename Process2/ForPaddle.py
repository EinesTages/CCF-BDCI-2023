import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer

tokenizer = AutoTokenizer.from_pretrained(r"B1/")
out = []
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [label for label in df['label']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(r'B1/')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 24)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        out.concat(pooled_output.tolist())
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


import pandas as pd
test_data = pd.read_csv('Data/Children.csv')


# 测试模型
def evaluate(model, test_data):
    i = 0
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=16, shuffle=False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        model = model.to(device)

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            print(i)
            i = i + 1


# 用测试数据集进行测试
test_data['label'] = [0 for _ in range(len(test_data['node_id']))]
model = torch.load("./checkpoint/8.pth")
evaluate(model, test_data)
np.save("./Graph_Data/feature.npy", np.array(out))
