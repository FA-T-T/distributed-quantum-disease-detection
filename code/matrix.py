from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
def plot_matrix1(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None, save_path=None):
    # 利用 sklearn 中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵 
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 设置图形尺寸，增加图像的长度
    plt.figure(figsize=(6, 6))  # 你可以根据需要调整宽度和高度

    # 画图，如果希望改变颜色风格，可以改变此部分的 cmap=pl.get_cmap('Blues') 处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在 x 轴坐标上，并倾斜 45 度
    plt.yticks(num_local, axis_labels)  # 将标签印在 y 轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 将百分比打印在相应的格子内，大于 thresh 的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            percentage = int(cm[i][j] * 100 + 0.5)
            plt.text(j, i, f"{percentage}%",
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    # 显示
    plt.show()

def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None, save_path=None, cell_font_size=12):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        labels_name: 标签名称列表
        title: 图像标题
        thresh: 阈值，用于确定文本颜色(大于阈值为白色，小于为黑色)
        axis_labels: 坐标轴标签
        save_path: 图像保存路径
        cell_font_size: 混淆矩阵格子内的字体大小，默认12
    """
    # 利用 sklearn 中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵 
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 设置图形尺寸
    plt.figure(figsize=(6, 6))

    # 画图
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)
    
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
        
    plt.xticks(num_local, axis_labels, rotation=45)
    plt.yticks(num_local, axis_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 将百分比打印在相应的格子内，增加字体大小参数
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            percentage = int(cm[i][j] * 100 + 0.5)
            plt.text(j, i, f"{percentage}%",
                    ha="center", va="center",
                    fontsize=cell_font_size,  # 新增参数，控制格子内字体大小
                    color="white" if cm[i][j] > thresh else "black")

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        
    # 显示
    plt.show()