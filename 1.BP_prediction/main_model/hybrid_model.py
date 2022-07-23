import torch
import torch.nn as nn
from running_scripts.MyDataset import MainDataset
import torch.optim as optim
from running_scripts import training_starter
import os


class Hybrid_model(nn.Module):  # 继承torch.nn.Module类
    def __init__(self, output_size=2):
        super(Hybrid_model, self).__init__()  # 获得Net类的超类（父类）的构造方法
        # 定义神经网络的每层结构形式
        # 各个层的信息都是Net类对象的属性
        self.hidden1 = nn.Linear(21, 128, bias=True)  # 隐藏层线性输出
        self.hidden2 = nn.Linear(128, 64, bias=True)  # 隐藏层线性输出
        self.predict = nn.Linear(64, output_size, bias=True)  # 输出层线性输出

    # 将各层的神经元搭建成完整的神经网络的前向通路
    def forward(self, seq, ft):
        x = self.hidden1(ft)
        x = torch.relu(x)  # 对隐藏层的输出进行relu激活
        x = self.hidden2(x)
        x = torch.relu(x)  # 对隐藏层的输出进行relu激活
        x = self.predict(x)
        return x


if __name__ == "__main__":
    MODEL_NAME = 'simple_MLP'
    user_id = '055'
    # targetBP = ['SBP', 'DBP']
    targetBP = ['SBP']

    # 配置文件路径
    RESULT_ROOT = r'G:\PythonPro\BP_prediction\result'
    best_model_param = os.path.join(RESULT_ROOT, 'params', user_id, MODEL_NAME + '_param.pkl')
    # training配置
    window_size = 1000
    EPOCH_NUM = 20
    LR = 0.005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = MLP_model(output_size=len(targetBP)).to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)  # 传入网络参数和学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = nn.MSELoss(reduction='mean')  # 最小均方误差
    # 加载数据集
    train_dataset = MainDataset('train', user_id, window_size, targetBP)
    vali_dataset = MainDataset('valid', user_id, window_size, targetBP)
    training_starter.model_training(MODEL_NAME, user_id, net, optimizer, scheduler, EPOCH_NUM, loss_function,
                                    train_dataset, vali_dataset, targetBP)

    test_dataset = MainDataset('test', user_id, window_size, targetBP)
    training_starter.model_testing(MODEL_NAME, user_id, net, test_dataset, targetBP)
