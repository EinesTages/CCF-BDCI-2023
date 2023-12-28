import csv
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("B2")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [label for label in df['label']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=768,
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


import pandas as pd

bbc_text_df = pd.read_csv('./Data/train.csv')
test_data = pd.read_csv('Data/test.csv')
# bbc_text_df.head()
df = pd.DataFrame(bbc_text_df)
np.random.seed(112)
df = df.sample(frac=1, random_state=21412)
df_train = df
print(len(df_train))

# 构建模型
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained("B2", num_labels=24).to("cuda")

    def forward(self, input_id, mask):
        final_layer = self.bert(input_ids=input_id, attention_mask=mask).logits
        return final_layer

from torch.optim import Adam
from tqdm import tqdm


def train(model, train_data, learning_rate, epochs):
    # 通过Dataset类获取训练和验证集
    train = Dataset(train_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=24, shuffle=True)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:4" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.to(device)
        criterion = criterion.to(device)
    # 开始进入训练循环
    for epoch_num in range(epochs):

        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        print(f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} ''')
        if epoch_num >= 4:
            torch.save(model, './checkpoint/{}.pt'.format(epoch_num + 1))


EPOCHS = 25
model = BertClassifier()
LR = 2e-5
train(model, df_train, LR, EPOCHS)
preds = []


# 测试模型
def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:4" if use_cuda else "cpu")
    if use_cuda:
        model = model.to(device)

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            preds.append(output.item)


# 用测试数据集进行测试
test_data['label'] = [0 for _ in range(len(test_data['node_id']))]
evaluate(model, test_data)
sub = pd.DataFrame()
sub['node_id'] = list(test_data['node_id'])
sub['label'] = preds
sub.to_csv('./Result/submission_bert.csv', index=False)
