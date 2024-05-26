import torch
import torch.optim as optim 
from transModel import DeformationNetworkSeparate
from DeformationLoss import DeformationLoss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 创建网络实例
model = DeformationNetworkSeparate().to(device)

# 检查模型是否在 GPU 上
if next(model.parameters()).is_cuda:
    print("Model is on GPU")
else:
    print("Model is on CPU")

# 创建损失函数实例
loss_fn = DeformationLoss(position_weight=1.0, quaternion_weight=1.0).to(device)

# 创建优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟数据
x = torch.tensor([0.1, 0.2, 0.3], device=device)
q = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
t = 1.0
target_x = torch.tensor([0.2, 0.3, 0.4], device=device)
target_q = torch.tensor([0.2, 0.3, 0.4, 0.5], device=device)

# 检查输入数据是否在 GPU 上
if x.is_cuda and q.is_cuda and target_x.is_cuda and target_q.is_cuda:
    print("Data is on GPU")
else:
    print("Data is on CPU")

# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    predicted_x, predicted_q = model(x, q, t)

    # 检查预测值是否在 GPU 上
    if predicted_x.is_cuda and predicted_q.is_cuda:
        print(f"Epoch {epoch+1}: Predictions are on GPU")
    else:
        print(f"Epoch {epoch+1}: Predictions are on CPU")

    # 计算损失
    loss = loss_fn(predicted_x, predicted_q, target_x, target_q)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")
