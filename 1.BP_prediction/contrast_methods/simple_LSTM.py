import torch
import torch.nn as nn
from running_scripts.MyDataset import MainDataset
import torch.optim as optim
from running_scripts import training_starter
import os


class LSTM_model(nn.Module):  # 继承torch.nn.Module类
    def __init__(self, h=16, output_size=2):
        super(LSTM_model, self).__init__()  # 获得Net类的超类（父类）的构造方法
        # 定义神经网络的每层结构形式
        # 各个层的信息都是Net类对象的属性
        self.h = h
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=h, batch_first=True)
        self.hidden1 = nn.Linear(h, 128, bias=True)  # 隐藏层线性输出
        self.predict = nn.Linear(128, output_size, bias=True)  # 输出层线性输出

    # 将各层的神经元搭建成完整的神经网络的前向通路
    def forward(self, seq, ft):
        seq = torch.reshape(seq, [seq.shape[0], seq.shape[1], 1])
        x, _ = self.lstm1(seq)
        x = torch.reshape(x[:, -1, :], [-1, self.h])
        x = torch.relu(x)  # 对隐藏层的输出进行relu激活
        x = self.hidden1(x)
        x = torch.relu(x)  # 对隐藏层的输出进行relu激活
        x = self.predict(x)
        return x


if __name__ == "__main__":
    MODEL_NAME = 'simple_LSTM'
    user_id = '055'
    # targetBP = ['SBP', 'DBP']
    targetBP = ['SBP']

    # training配置
    window_size = 1000
    EPOCH_NUM = 20
    LR = 0.005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = LSTM_model(h=32, output_size=len(targetBP)).to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)  # 传入网络参数和学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = nn.MSELoss(reduction='mean')  # 最小均方误差
    # 加载数据集
    nrows = None
    train_dataset = MainDataset('train', user_id, window_size, targetBP, nrows=nrows)
    vali_dataset = MainDataset('valid', user_id, window_size, targetBP, nrows=nrows)
    training_starter.model_training(MODEL_NAME, user_id, net, optimizer, scheduler, EPOCH_NUM, loss_function,
                                    train_dataset, vali_dataset, targetBP)

    test_dataset = MainDataset('test', user_id, window_size, targetBP, nrows=nrows)
    training_starter.model_testing(MODEL_NAME, user_id, net, test_dataset, targetBP)
