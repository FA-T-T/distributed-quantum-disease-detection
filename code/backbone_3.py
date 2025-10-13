import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
# from VGGNet import vgg16_net
# from QNN import QNN
# from QNN_copy import QNN
from mobilnet import MobileNetV2
from mps3 import QNN

# Classic Network: VGG16
# Quantum Network: 

# n_layers = 1 # 电路层数
# n_qubits = 4 # vgg输出等于量子比特个数
# # qml.device 指定后端模拟器和量子比特数，这里default.qubit模拟纯态qubit，也可以用default.mixed模拟混合态qubit，对应含噪电路

class QCNet(nn.Module):
    def __init__(self):
        super(QCNet, self).__init__()
        self.qubits=8
        self.nclass=3
        # # 经典神经网络
        # self.CModel = MobileNetV2() 
        # 经典神经网络
        self.CModel = MobileNetV2()
        # self.CModel.load_state_dict(cmodel_state_dict, strict=False)
        # for param in self.CModel.parameters():
        #     param.requires_grad = False  # 冻结经典模型参数


        # 全连接参数配置
        self.fc1 = nn.Linear(7, self.qubits) 
        # 量子神经网络
        self.QModel = QNN()
        # 全连接层
        self.fc2 = nn.Linear(32, self.nclass)
        # self.fc2 =nn.Linear(self.qubits, 3)

        # self.lr1 = nn.LeakyReLU(0.1)



    def forward(self, x):
        # Forward Propagation
        x = self.CModel(x)
        # x = self.fc1(x) 
        # print(2)      
        x = self.QModel(x)
        
        # print(3)
        x = self.fc2(x)
        # print(x)
        # print(4)
        # x=F.softmax(x,dim=1)
        return x
