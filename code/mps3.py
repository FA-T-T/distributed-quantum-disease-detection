import pennylane as qml
import torch.nn as nn
# import torch.nn.functional as F
import torch
import numpy as np
import os

# 文件夹中第4组的方案

# 卷积电路分割后不能出现纠缠现象！！！

# 问题：参数对应问题，参数维度等########################################
# 问题：数据输入网络问题，切割点在制备过程中是否还需要inputs########################################

# 只要跟切割电路有关，是否都可以用该方法复现？将所有与切割电路有关的输出线路，全部测量，根据切割点测量值，逐一恢复
# qml.qnn.TorchLayer层的返回值类型  tensor类型
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Quantum Network: 
n_layers = 1  # 电路层数
n_qubits_1 = 4  # 8+9 16+17比特？
n_qubits_2 = 5
# qml.device 指定后端模拟器和量子比特数，这里default.qubit模拟纯态qubit，也可以用default.mixed模拟 lightning.qubit CPU lightning.gpu gpu
dev_1 = qml.device('lightning.qubit', wires=n_qubits_1, batch_obs = True)
dev_2 = qml.device('lightning.qubit', wires=n_qubits_2, batch_obs = True)


# 构建编码电路
def embedding_circuit(inputs, n_qubits):
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)


# 构建卷积层
def conv_circuit_1(layer, n_qubits, weights):
    param_index = 0
    for i in range(n_qubits-1):

        qml.RZ(weights[layer, param_index], wires=(i + 1) % n_qubits)
        qml.CRZ(weights[layer, param_index+1], wires=[i, (i + 1) % n_qubits])

        qml.RY(weights[layer, param_index + 2], wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.RY(weights[layer, param_index + 3], wires=(i + 1) % n_qubits)
        qml.CRZ(weights[layer, param_index+4], wires=[i, (i + 1) % n_qubits])
        param_index += 5 

def conv_circuit_2(layer, n_qubits, weights):
    param_index = 0
    for i in range(n_qubits-1):
        qml.RZ(weights[layer, param_index], wires=(i + 1) % n_qubits)
        qml.CRZ(weights[layer, param_index+1], wires=[i, (i + 1) % n_qubits])

        qml.RY(weights[layer, param_index + 2], wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.RY(weights[layer, param_index + 3], wires=(i + 1) % n_qubits)
        qml.CRZ(weights[layer, param_index+4], wires=[i, (i + 1) % n_qubits])
        param_index += 5  

# 构建电路1（不包含测量操作，最后1bit作为切割点）(4bit)
def build_circuits1(inputs, weights, qubits):
    embedding_circuit(inputs, qubits)
    for layer in range(n_layers):
        conv_circuit_1(layer, qubits, weights)

# 构建电路2（不包含测量操作，最后1bit作为切割点）(5bit)
def build_circuits2(inputs, weights, qubits):
    embedding_circuit(inputs, qubits)
    for layer in range(n_layers):
        conv_circuit_2(layer, qubits, weights)


############################################################################
# 测量
@qml.qnode(dev_1, interface="torch")
def circuit_front_1(inputs, weights):
    build_circuits1(inputs, weights, n_qubits_1)
    # return qml.probs(wires=[i for i in range(n_qubits_1)]) # return numpy.array()
    return qml.probs(wires=n_qubits_1 - 1)


@qml.qnode(dev_1, interface="torch")
def circuit_front_2(inputs, weights):
    build_circuits1(inputs, weights, n_qubits_1)
    qml.Hadamard(wires=n_qubits_1 - 1)
    # return qml.probs(wires=[i for i in range(n_qubits_1)])
    return qml.probs(wires=n_qubits_1 - 1)


@qml.qnode(dev_1, interface="torch")
def circuit_front_3(inputs, weights):
    build_circuits1(inputs, weights, n_qubits_1)
    qml.RX(np.pi / 2, wires=n_qubits_1 - 1)
    # return qml.probs(wires=[i for i in range(n_qubits_1)])
    return qml.probs(wires=n_qubits_1 - 1)

############################################################################
# 制备
# 定义量子节点
@qml.qnode(dev_2, interface="torch")
def circuit_back_1(inputs, weights):
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_2(inputs, weights):
    qml.PauliX(wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_3(inputs, weights):
    qml.Hadamard(wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_4(inputs, weights):
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_5(inputs, weights):
    qml.RX(np.pi / 2, wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_6(inputs, weights):
    qml.RX(-np.pi / 2, wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


def calculate_probabilities(tensor):
    # 假设tensor的形状是 [batch_size, num_probs]
    # tensor中的偶数索引代表事件0，奇数索引代表事件1
    prob_event_0 = tensor[:, 0::2].sum(dim=1)  # 沿着第二维求和，取间隔为2的元素，开始于索引0
    prob_event_1 = tensor[:, 1::2].sum(dim=1)  # 同上，但开始于索引1

    # 将结果堆叠成一个新的tensor，形状为 [batch_size, 2]
    result_tensor = torch.stack([prob_event_0,prob_event_1], dim=1)
    return result_tensor


def combine_outputs(probabilities_front_1, probabilities_front_2, probabilities_front_3, output_back_1, output_back_2, output_back_3, 
                    output_back_4, output_back_5, output_back_6):
    # 将所有后端输出转为列表
    res2 = torch.stack([output_back_1, output_back_2, output_back_3,
                        output_back_4, output_back_5, output_back_6], dim=1) # torch.Size([2**n_qubits_2]) => torch.Size([batch_size, 6, 2**n_qubits_2])

    result_1 = torch.cat((probabilities_front_1, probabilities_front_2, probabilities_front_3), dim=1) # torch.Size([batch_size, 6])

    # 计算nz, nx, ny
    nz = result_1[:, 0] - result_1[:, 1] # torch.Size([batch_size])
    nx = result_1[:, 2] - result_1[:, 3]
    ny = result_1[:, 4] - result_1[:, 5]

    # 计算c系数
    c0 = (1 + nz) / 2 # torch.Size([batch_size])
    c1 = (1 - nz) / 2
    c2 = nx / 2
    c3 = -c2
    c4 = ny / 2
    c5 = -c4

    c_list = torch.stack([c0, c1, c2, c3, c4, c5], dim=1) # torch.Size([batch_size, 6])

    # 计算最终结果
    # result_list = torch.zeros_like(res2[0])  # 假设所有输出批次大小和维度相同
    result_list = torch.zeros_like(output_back_1) # torch.Size([batch_size, 2**n_qubits_2])

    for j in range(c_list.shape[1]):
        result_list += c_list[:, j].unsqueeze(1) * res2[:, j, :]

    return result_list


class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        
        # 定义共享权重张量 - 使用正确的初始化范围 [-π/4, π/4] 而不是 [0, 1)
        # 这对量子旋转门更合适，有助于训练稳定性
        self.shared_weights_1 = nn.Parameter(torch.empty((n_layers, (n_qubits_1 - 1) * 5)), requires_grad=True)
        nn.init.uniform_(self.shared_weights_1, -np.pi/4, np.pi/4)
        
        weight_shapes_1 = {"weights": (n_layers, (n_qubits_1 - 1) * 5)}
        self.QLayer_front_1 = qml.qnn.TorchLayer(circuit_front_1, weight_shapes_1)
        self.QLayer_front_2 = qml.qnn.TorchLayer(circuit_front_2, weight_shapes_1)
        self.QLayer_front_3 = qml.qnn.TorchLayer(circuit_front_3, weight_shapes_1)
        
        # 使用.data共享权重 - 注意这会使每个层使用相同的权重参数
        self.QLayer_front_1.weights.data = self.shared_weights_1
        self.QLayer_front_2.weights.data = self.shared_weights_1
        self.QLayer_front_3.weights.data = self.shared_weights_1

        self.shared_weights_2 = nn.Parameter(torch.empty((n_layers, (n_qubits_2 - 1) * 5)), requires_grad=True)
        nn.init.uniform_(self.shared_weights_2, -np.pi/4, np.pi/4)
        
        weight_shapes_2 = {"weights": (n_layers, (n_qubits_2 - 1) * 5)}
        self.QLayer_back_1 = qml.qnn.TorchLayer(circuit_back_1, weight_shapes_2)
        self.QLayer_back_2 = qml.qnn.TorchLayer(circuit_back_2, weight_shapes_2)
        self.QLayer_back_3 = qml.qnn.TorchLayer(circuit_back_3, weight_shapes_2)
        self.QLayer_back_4 = qml.qnn.TorchLayer(circuit_back_4, weight_shapes_2)
        self.QLayer_back_5 = qml.qnn.TorchLayer(circuit_back_5, weight_shapes_2)
        self.QLayer_back_6 = qml.qnn.TorchLayer(circuit_back_6, weight_shapes_2)
        
        # 使用.data共享权重
        self.QLayer_back_1.weights.data = self.shared_weights_2
        self.QLayer_back_2.weights.data = self.shared_weights_2
        self.QLayer_back_3.weights.data = self.shared_weights_2
        self.QLayer_back_4.weights.data = self.shared_weights_2
        self.QLayer_back_5.weights.data = self.shared_weights_2
        self.QLayer_back_6.weights.data = self.shared_weights_2

    def forward(self, inputs): # inputs = stack([batch_size, 8], dim=0)
        # 注意：TorchLayer不支持原生批处理，需要逐样本处理
        # 这是PennyLane TorchLayer的限制，不是设计缺陷
        
        output_front_1=[]
        output_front_2=[]
        output_front_3=[]

        output_back_1=[]
        output_back_2=[]
        output_back_3=[]
        output_back_4=[]
        output_back_5=[]
        output_back_6=[]
        
        for i in inputs:
            output_front_1.append(self.QLayer_front_1(i[:n_qubits_1]))
            output_front_2.append(self.QLayer_front_2(i[:n_qubits_1]))
            output_front_3.append(self.QLayer_front_3(i[:n_qubits_1]))
            output_back_1.append(self.QLayer_back_1(i[n_qubits_1-1:]))
            output_back_2.append(self.QLayer_back_2(i[n_qubits_1-1:]))
            output_back_3.append(self.QLayer_back_3(i[n_qubits_1-1:]))
            output_back_4.append(self.QLayer_back_4(i[n_qubits_1-1:]))
            output_back_5.append(self.QLayer_back_5(i[n_qubits_1-1:]))
            output_back_6.append(self.QLayer_back_6(i[n_qubits_1-1:]))
        
        output_front_1 = torch.stack(output_front_1, dim=0)
        output_front_2 = torch.stack(output_front_2, dim=0)
        output_front_3 = torch.stack(output_front_3, dim=0)
        output_back_1 = torch.stack(output_back_1, dim=0)
        output_back_2 = torch.stack(output_back_2, dim=0)
        output_back_3 = torch.stack(output_back_3, dim=0)
        output_back_4 = torch.stack(output_back_4, dim=0)
        output_back_5 = torch.stack(output_back_5, dim=0)
        output_back_6 = torch.stack(output_back_6, dim=0)

        combined_outputs = combine_outputs(output_front_1, output_front_2, output_front_3,
                                            output_back_1, output_back_2, output_back_3,
                                            output_back_4, output_back_5, output_back_6)

        return combined_outputs

