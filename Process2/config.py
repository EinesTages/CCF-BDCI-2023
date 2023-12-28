import torch


class Config:
    mode = 'MachineLearning'  # 模式,应支持传统机器学习方法或深度学习方法(更具体到语言模型还是图模型)
    pre_process = 'TF-IDF'  # 文本特征预处理方式,支持 TD-IDF or Word2Vec
    model = 'LightGBM'  # 模型选用,应可支持SVM,各种bert,以及图神经网络
    dataset_path = 'Data'  # 数据集保存路径
    dataset_name = 'Children'  # 数据集名称，理应可选Children，train，test
    epochs = 10000  # 训练轮数
    lr = 0.01  # 学习率/保证Bert和正常深度学习方法不一致
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
