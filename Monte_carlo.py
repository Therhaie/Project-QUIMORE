#!/usr/bin/env python

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from examples.spde.solver import SteadyStateHeat2DSolver  # Ensure this path is correct
from scripts.surrogate_pytorch import DeepUQSurrogate  # Ensure surrogate_pytorch has PyTorch model

# Model path and device setup
model_path = os.path.join(os.path.dirname(__file__), "models", "iteration_model_4096.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Model setup
nx = ny = 32
D = nx * ny
L = 3  # Number of encoding layers
d = 128  # Encoding dimension

model = DeepUQSurrogate(D=D, L=L, d=d, output_size=D).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the dataset
data_path = r"C:\python_workspace\3A\ProjetSOIA\deep-uq-paper-master\data\file_4096\test_file\data_rbf_samples_40960_nx_32_ny_32_8_4096.npz"
data = np.load(data_path)
inputs = data['inputs'].reshape(-1, nx * ny)
outputs = data['outputs'].reshape(-1, nx * ny)

# Monte Carlo parameters
num_samples = 10  # Number of Monte Carlo realizations
mean_k, std_k = 0.5, 0.1  # Mean and standard deviation of random parameter (example)

# Parameters for batching
batch_size = 10  # Adjust based on available GPU memory
num_batches = num_samples // batch_size

# Generate random inputs
monte_carlo_inputs = []
for _ in range(num_samples):
    random_variation = np.random.normal(loc=mean_k, scale=std_k, size=(nx, ny))
    monte_carlo_inputs.append(inputs + random_variation.flatten())

monte_carlo_inputs = np.array(monte_carlo_inputs)

# Convert to tensors
monte_carlo_inputs = torch.from_numpy(monte_carlo_inputs).float().to(device)
monte_carlo_predictions = []

# Perform predictions for all Monte Carlo realizations
with torch.no_grad():
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        
        # Select a batch and move to GPU
        input_batch = torch.from_numpy(monte_carlo_inputs[start:end].numpy()).float().to(device)
        
        # Perform predictions for the batch
        predictions = model(input_batch).reshape(batch_size, nx, ny).cpu().numpy()
        monte_carlo_predictions.extend(predictions)


monte_carlo_predictions = np.array(monte_carlo_predictions)

# Compute uncertainty metrics
mean_prediction = np.mean(monte_carlo_predictions, axis=0)
variance_prediction = np.var(monte_carlo_predictions, axis=0)
std_dev_prediction = np.sqrt(variance_prediction)

# Visualize uncertainty
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
im1 = axes[0].imshow(mean_prediction, cmap="viridis")
axes[0].set_title("Mean Prediction")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(std_dev_prediction, cmap="inferno")
axes[1].set_title("Uncertainty (Standard Deviation)")
plt.colorbar(im2, ax=axes[1])

# Confidence intervals
lower_bound = mean_prediction - 1.96 * std_dev_prediction
upper_bound = mean_prediction + 1.96 * std_dev_prediction

im3 = axes[2].imshow(upper_bound - lower_bound, cmap="plasma")
axes[2].set_title("95% Confidence Interval Width")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()

# Compute MSE and SSIM for Monte Carlo realizations if ground truth is available
if 'outputs' in data:
    mse_values = []
    ssim_values = []

    ground_truth_outputs = outputs.reshape(-1, nx, ny)
    for i in range(num_samples):
        truth = ground_truth_outputs[i]
        pred = monte_carlo_predictions[i]
        mse = np.mean((truth - pred) ** 2)
        similarity = ssim(truth, pred, data_range=1.0)
        mse_values.append(mse)
        ssim_values.append(similarity)

    print(f"Monte Carlo Average MSE: {np.mean(mse_values):.4f}")
    print(f"Monte Carlo Average SSIM: {np.mean(ssim_values):.4f}")
