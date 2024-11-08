#!/usr/bin/env python

import argparse
import numpy as np
import os
from examples.spde.solver import SteadyStateHeat2DSolver  # Ensure this path is correct for your setup
import GPy
import torch
from scripts.surrogate_pytorch import DeepUQSurrogate  # Ensure surrogate_pytorch has PyTorch model
import matplotlib.pyplot as plt
import time

# Argument parsing for flexibility
parser = argparse.ArgumentParser(description="Run heat equation solver with random field inputs.")
parser.add_argument('-N', type=int, default=10, help="Number of samples")
parser.add_argument('-nx', type=int, default=32, help="Number of grid cells in x direction")
parser.add_argument('-ny', type=int, default=32, help="Number of grid cells in y direction")
parser.add_argument('-lx', type=float, default=0.02, help="Lengthscale x")
parser.add_argument('-ly', type=float, default=0.02, help="Lengthscale y")
parser.add_argument('-var', type=float, default=1.0, help="Variance of random field")
parser.add_argument('-k', type=str, default="rbf", help="Kernel type for GP model")
args = parser.parse_args()

# Define the kernel and other parameters
kernels = {'rbf': GPy.kern.RBF, 'exp': GPy.kern.Exponential, 
           'mat32': GPy.kern.Matern32, 'mat52': GPy.kern.Matern52}

num_samples = args.N
nx = args.nx
ny = args.ny
lx = args.lx
ly = args.ly
variance = args.var
kernel_type = args.k
assert kernel_type in kernels
kernel = kernels[kernel_type](input_dim=2, lengthscale=[lx, ly], variance=variance, ARD=True)

# Initialize solver
solver = SteadyStateHeat2DSolver(nx=nx, ny=ny)
cellcenters = solver.mesh.cellCenters.value.T

# Placeholder arrays to save input/output samples
inputs = np.zeros((num_samples, nx, ny))
outputs = np.zeros((num_samples, nx, ny))

# Run the solver and generate data
for i in range(num_samples):
    print(f"Generating sample {i + 1}/{num_samples}")
    
    # Generate random field sample
    mean = np.zeros((cellcenters.shape[0], 1))
    C = kernel.K(cellcenters) + 1e-6 * np.eye(cellcenters.shape[0])
    L = np.linalg.cholesky(C)
    random_field = mean + np.dot(L, np.random.randn(cellcenters.shape[0], 1))
    sample = np.exp(random_field[:, 0])

    # Set up and solve
    solver.set_coeff(C=sample)
    solver.set_source(source=np.zeros_like(sample))  # Assuming zero source
    solver.solve()

    # Save input-output pairs
    inputs[i] = random_field.reshape((nx, ny))
    outputs[i] = solver.phi.value.reshape((nx, ny))

# Save data
# Get the absolute path of the current script
path_file = os.path.abspath(__file__)

# Create a 'data' directory in the same directory as the current script
data_directory = os.path.join(os.path.dirname(path_file), "data")
os.makedirs(data_directory, exist_ok=True)

# Create the full path to the output file
output_file = os.path.join(data_directory, f"data_{kernel_type}_samples_500.npz")

# Save the data
np.savez(output_file, inputs=inputs, outputs=outputs)
print(f"Data saved to {output_file}")
print("Output file saved in:", os.path.abspath(output_file))

#########################################################################################################################
# # TRAINING PART

# Load the generated dataset
data_path = os.path.join(os.path.dirname(__file__), "data", "data_rbf_samples.npz")
print(data_path)
data = np.load(data_path)

# Reshape inputs and outputs for training (flat vectors)
# inputs = data['inputs'].reshape(-1, nx * ny)  # Flatten for training (num_samples, nx * ny)
# outputs = data['outputs'].reshape(-1, nx * ny)  # Flatten for training (num_samples, nx * ny)
inputs = data['inputs'].reshape(-1, nx * ny)
outputs = data['outputs'].reshape(-1, nx * ny)

# # Convert inputs and outputs to PyTorch tensors and move to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# inputs = torch.from_numpy(inputs).float().to(device)
# outputs = torch.from_numpy(outputs).float().to(device)
# Convert to PyTorch tensors and send to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = torch.from_numpy(inputs).float().to(device)
outputs = torch.from_numpy(outputs).float().to(device)


# Convert inputs and outputs to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

# Ensure that the output shape matches the model's output shape
assert inputs.shape[1] == nx * ny, f"Expected input shape ({nx * ny}), but got {inputs.shape[1]}"
assert outputs.shape[1] == nx * ny, f"Expected output shape ({nx * ny}), but got {outputs.shape[1]}"

# Initialize and train the surrogate model
D = nx * ny  # Dimensionality of input
print('value of D= nx*ny',D)
L = 3        # Number of encoding layers
d = 128      # Encoding dimension

surrogate = DeepUQSurrogate(D=D, L=L, d=d, output_size=1024).to(device) # be careful with the output_size shape of the 

# Define optimizer and loss function
optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Training parameters
epochs = 500
batch_size = 32
loss_values = []

# Track training time
start_time = time.time()

# Training loop with loss tracking for plotting
for epoch in range(epochs):
    for i in range(0, inputs.size(0), batch_size):
        x_batch = inputs[i:i + batch_size]
        y_batch = outputs[i:i + batch_size]
        
        # Forward pass
        y_pred = surrogate(x_batch)
        
        # Compute loss
        loss = criterion(y_pred, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Log and measure time every 50 epochs
    if (epoch + 1) % 50 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        loss_val = criterion(surrogate(inputs), outputs).item()
        
        # Record loss and reset start time
        print(f"Epoch {epoch+1}, Loss: {loss_val}, Time for last 50 epochs: {elapsed_time:.2f} seconds")
        loss_values.append(loss_val)
        start_time = time.time()  # Reset timer for the next 50 epochs

print("Training complete.")

# Define the path where the model will be saved
model_path = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(model_path, exist_ok=True)
model_path = os.path.join(model_path, "trained_model.pth") 

# Need to modify the name of the output model to be able to extract importante information if needed to be used / reloaed later

# Save the model
torch.save(surrogate.state_dict(), model_path)
print(f"Model saved to {model_path}")


# Plot the evolution of the loss
plt.figure(figsize=(10, 5))
plt.plot(range(0, epochs, 50), loss_values, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolution of Training Loss")
plt.legend()
plt.show()

# Plot predictions vs ground truth
# Generate predictions for a subset of data (e.g., the first 5 samples)
with torch.no_grad():
    predictions = surrogate(inputs[:5]).reshape(-1, nx, ny)
    ground_truth = outputs[:5].reshape(-1, nx, ny)

# Plot ground truth vs predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    # Ground truth
    axes[0, i].imshow(ground_truth[i].cpu().numpy(), cmap="viridis") # cpu is used to move the tensor to cpu
    axes[0, i].set_title(f"Ground Truth {i+1}")
    axes[0, i].axis("off")
    
    # Prediction
    axes[1, i].imshow(predictions[i].cpu().numpy(), cmap="viridis") # cpu is used to move the tensor to cpu
    axes[1, i].set_title(f"Prediction {i+1}")
    axes[1, i].axis("off")

plt.suptitle("Ground Truth vs Predictions")
plt.tight_layout()


# Test the performance of the model reloaded

# Initialize the model (with the same structure) and load the state dictionary
surrogate_load = DeepUQSurrogate(D=nx * ny, L=3, d=128, output_size=1024).to(device)
surrogate_load.load_state_dict(torch.load(model_path))
surrogate_load.eval()  # Set the model to evaluation mode
print("Model loaded successfully.")

# Generate predictions for a subset of data (e.g., the first 5 samples)
with torch.no_grad():
    predictions = surrogate_load(inputs[:5]).reshape(-1, nx, ny)
    ground_truth = outputs[:5].reshape(-1, nx, ny)

# Plot ground truth vs predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    # Ground truth
    axes[0, i].imshow(ground_truth[i].cpu().numpy(), cmap="viridis")
    axes[0, i].set_title(f"Ground Truth {i+1}")
    axes[0, i].axis("off")
    
    # Prediction
    axes[1, i].imshow(predictions[i].cpu().numpy(), cmap="viridis")
    axes[1, i].set_title(f"Prediction {i+1}")
    axes[1, i].axis("off")


plt.suptitle("Ground Truth vs Predictions reloaded model")
plt.tight_layout()
plt.show()

