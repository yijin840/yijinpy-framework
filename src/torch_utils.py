import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 输入2维，隐藏层10个神经元
        self.fc2 = nn.Linear(10, 2)  # 输出2维，二分类

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def torch_start():
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 1. 数据准备
    # 随机生成1000个样本，每个样本有2个特征
    x_data = torch.randn(1000, 2)
    y_data = (x_data[:, 0] * x_data[:, 1] > 0).long()  # 假设根据特征乘积的符号进行分类

    # 转换为 PyTorch 数据集和数据加载器
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # 2. 定义模型
    model = SimpleNN().to(device)
    # 3. 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 使用 Adam 优化器

    # 4. 模型训练
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")
    # 5. 模型评估
    model.eval()
    with torch.no_grad():
        sample_data = torch.tensor(
            [[0.5, -0.5], [-0.5, 0.5], [1.0, 1.0]], device=device
        )
        predictions = model(sample_data)
        predicted_labels = torch.argmax(predictions, dim=1)
        print("Sample predictions:", predicted_labels.cpu().numpy())


def loadModel(model_path):
    pass


def loadModelParams(model_params_path):
    pass


def saveModel(model_path):
    pass


def saveModelParams(model_params_path):
    pass


def trainModel():
    pass


def loadDataStore(ds_path):
    pass


def saveDataStore(ds_path):
    pass
