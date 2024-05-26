import torch
import torch.optim as optim 
from transModel import DeformationNetworkSeparate, DeformationNetworkConnected
from DeformationLoss import DeformationLoss

'''
The test involves:
1. Initializing the model and loss function.
2. Creating an optimizer for model training.
3. Simulating input and target data.
4. Performing a training loop to optimize the model parameters based on the loss.
5. Checking if the computations are performed on the GPU or CPU.
'''

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize the model
# model = DeformationNetworkSeparate().to(device)
model = DeformationNetworkConnected().to(device)

# Check if the model is on GPU
if next(model.parameters()).is_cuda:
    print("Model is on GPU")
else:
    print("Model is on CPU")

# Initialize the loss function
loss_fn = DeformationLoss(position_weight=0.5, quaternion_weight=0.5).to(device)

# Create an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simulate input data
x = torch.tensor([0.1, 0.2, 0.3], device=device)
q = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
t = 1.0
target_x = torch.tensor([0.2, 0.3, 0.4], device=device)
target_q = torch.tensor([0.2, 0.3, 0.4, 0.5], device=device)

# Check if the input data is on GPU
if x.is_cuda and q.is_cuda and target_x.is_cuda and target_q.is_cuda:
    print("Data is on GPU")
else:
    print("Data is on CPU")

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    predicted_x, predicted_q = model(x, q, t)

    # Check if the predictions are on GPU
    if predicted_x.is_cuda and predicted_q.is_cuda:
        print(f"Epoch {epoch+1}: Predictions are on GPU")
    else:
        print(f"Epoch {epoch+1}: Predictions are on CPU")

    # Compute the loss
    loss = loss_fn(predicted_x, predicted_q, target_x, target_q)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

