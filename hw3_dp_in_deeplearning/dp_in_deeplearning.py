import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

# 定義超參數
batch_size = 64
learning_rate = 0.01
epochs = 10
max_grad_norm = 1.2  # 梯度裁剪的範數

# 定義 MNIST 資料集的轉換
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加載 MNIST 資料集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 實例化模型、損失函數和優化器
model = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 8, 2, padding=3),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(2, 1),
                            torch.nn.Conv2d(16, 32, 4, 2),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(2, 1),
                            torch.nn.Flatten(),
                            torch.nn.Linear(32 * 4 * 4, 32),
                            torch.nn.ReLU(),
                            torch.nn.Linear(32, 10))

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

criterion = nn.CrossEntropyLoss()

# 創建 PrivacyEngine 並將其應用到模型、優化器和數據加載器
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=max_grad_norm,
)

# 訓練迴圈
def train(model, train_loader, optimizer, epoch, device, delta):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")


for epoch in range(1, 11):
    train(model, train_loader, optimizer, epoch, device="cpu", delta=1e-5)
