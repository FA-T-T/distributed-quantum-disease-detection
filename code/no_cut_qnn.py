import pennylane as qml
import torch.nn as nn
# import torch.nn.functional as F
import torch
import numpy as np
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Quantum Network: 
n_layers = 1  # 电路层数
n_qubits_1 = 8 # 8+9 16+17比特？

# qml.device 指定后端模拟器和量子比特数，这里default.qubit模拟纯态qubit，也可以用default.mixed模拟 lightning.qubit CPU lightning.gpu gpu
dev_1 = qml.device('lightning.qubit', wires=n_qubits_1, batch_obs = True)



# 构建编码电路
def embedding_circuit(inputs, n_qubits):
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)


# 构建卷积层
def conv_circuit_1(layer, n_qubits, weights):
    param_index = 0
    n_qubits1=n_qubits//2
    for i in range(n_qubits1):
        qml.RZ(weights[layer, param_index], wires=(i + 1) % n_qubits1)
        qml.CRZ(weights[layer, param_index+1], wires=[i, (i + 1) % n_qubits1])
        qml.RY(weights[layer, param_index + 2], wires=(i + 1) % n_qubits1)
        qml.CNOT(wires=[i, (i + 1) % n_qubits1])
        qml.RY(weights[layer, param_index + 3], wires=(i + 1) % n_qubits1)
        qml.CRZ(weights[layer, param_index+4], wires=[i, (i + 1) % n_qubits1])
        param_index += 5 
    n_qubits2=n_qubits//2-1
    for i in range(n_qubits//2-1,n_qubits-1):
        qml.RZ(weights[layer, param_index], wires=i + 1)
        qml.CRZ(weights[layer, param_index+1], wires=[i, i + 1])
        qml.RY(weights[layer, param_index + 2], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
        qml.RY(weights[layer, param_index + 3], wires=i + 1)
        qml.CRZ(weights[layer, param_index+4], wires=[i, i + 1])
        param_index += 5 
    for i in range(n_qubits-1,n_qubits):
        qml.RZ(weights[layer, param_index], wires=n_qubits2)
        qml.CRZ(weights[layer, param_index+1], wires=[i, n_qubits2])
        qml.RY(weights[layer, param_index + 2], wires=n_qubits2)
        qml.CNOT(wires=[i, n_qubits2])
        qml.RY(weights[layer, param_index + 3], wires=n_qubits2)
        qml.CRZ(weights[layer, param_index+4], wires=[i,n_qubits2])
        param_index += 5 




# 构建电路1（不包含测量操作，最后1bit作为切割点）(4bit)
def build_circuits1(inputs, weights, qubits):
    embedding_circuit(inputs, qubits)
    for layer in range(n_layers):
        conv_circuit_1(layer, qubits, weights)

############################################################################
# 测量
@qml.qnode(dev_1, interface="torch")
def circuit_front_1(inputs, weights):
    build_circuits1(inputs, weights, n_qubits_1)

    return qml.probs(wires=[i for i in range(n_qubits_1//2-1, n_qubits_1)])  # return numpy.array()
    # return qml.probs(op=qml.Hermitian(H, wires=[i for i in range(n_qubits_1//2-1, n_qubits_1)]))

def calculate_probabilities(tensor):
    # 假设tensor的形状是 [batch_size, num_probs]
    # tensor中的偶数索引代表事件0，奇数索引代表事件1
    prob_event_0 = tensor[:, 0::2].sum(dim=1)  # 沿着第二维求和，取间隔为2的元素，开始于索引0
    prob_event_1 = tensor[:, 1::2].sum(dim=1)  # 同上，但开始于索引1

    # 将结果堆叠成一个新的tensor，形状为 [batch_size, 2]
    result_tensor = torch.stack([prob_event_0,prob_event_1], dim=1)
    return result_tensor


class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()

        weight_shapes_1 = {"weights": (n_layers, (n_qubits_1 + 1) * 5)} # 指定weights的形状，参数配置？
        self.QLayer_front_1 = qml.qnn.TorchLayer(circuit_front_1, weight_shapes_1)  # QNode ==> TorchLayer
    def forward(self, inputs): # inputs = stack(batch_size, x], dim=0)
        output_front_1=[]
        for i in inputs:
            output_front_1.append(self.QLayer_front_1(i[:n_qubits_1]))

        
        output_front_1 = torch.stack(output_front_1, dim=0)
        return output_front_1