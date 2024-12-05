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
from torchsummary import summary
import random
from skimage.metrics import structural_similarity as ssim

model_path = os.path.join(os.path.dirname(__file__), "models", "trained_model.pth")
print(model_path)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the surrogate model
nx = ny = 32
D = nx * ny
print('value of D = nx * ny:', D)
L = 3        # Number of encoding layers
d = 128      # Encoding dimension

model = DeepUQSurrogate(D=D, L=L, d=d, output_size=D).to(device)
model.load_state_dict(torch.load(model_path))

# Set the model into evaluation mode
model.eval()

# Load test data
data_path = rf"C:\python_workspace\3A\ProjetSOIA\deep-uq-paper-master\data\data_rbf_samples_2000_nx_32_ny_32.npz"
data = np.load(data_path)
inputs = data['inputs'].reshape(-1, nx * ny)
outputs = data['outputs'].reshape(-1, nx * ny)

# Split into training and testing sets
num_samples = inputs.shape[0]
test_size = int(0.8 * num_samples)
random_indices = np.array(random.sample(range(num_samples), test_size))
test_inputs = inputs[random_indices, :]
ground_truth_output = outputs[random_indices, :]

# Convert to tensors
test_inputs = torch.from_numpy(test_inputs).float().to(device)
ground_truth_output = torch.from_numpy(ground_truth_output).float().to(device)

# Evaluate model and analyze results
with torch.no_grad():
    test_predictions = model(test_inputs).reshape(-1, nx, ny)
    test_ground_truth = ground_truth_output.reshape(-1, nx, ny).to(device)

    # Select 5 random samples
    indices = np.random.choice(test_predictions.size(0), 5, replace=False)

    # Plot ground truth vs predictions for 5 random samples
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))  # Additional row for difference maps
    for idx, i in enumerate(indices):
        # Ground truth
        axes[0, idx].imshow(test_ground_truth[i].cpu().numpy(), cmap="viridis")
        axes[0, idx].set_title(f"Ground Truth {i+1}")
        axes[0, idx].axis("off")

        # Prediction
        axes[1, idx].imshow(test_predictions[i].cpu().numpy(), cmap="viridis")
        axes[1, idx].set_title(f"Prediction {i+1}")
        axes[1, idx].axis("off")

        # Difference Map
        diff = np.abs(test_ground_truth[i].cpu().numpy() - test_predictions[i].cpu().numpy())
        axes[2, idx].imshow(diff, cmap="coolwarm")
        axes[2, idx].set_title(f"Difference {i+1}")
        axes[2, idx].axis("off")

    plt.suptitle("Ground Truth vs Predictions with Difference Maps")
    plt.tight_layout()
    plt.show()

    # Compute metrics for all test samples
    mse_values = []
    ssim_values = []
    for i in range(test_predictions.size(0)):
        truth = test_ground_truth[i].cpu().numpy()
        pred = test_predictions[i].cpu().numpy()
        mse = np.mean((truth - pred) ** 2)
        similarity = ssim(truth, pred)
        mse_values.append(mse)
        ssim_values.append(similarity)

    # Print average metrics
    print(f"Average MSE: {np.mean(mse_values):.4f}")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")

    # Plot histogram of errors
    errors = test_ground_truth.cpu().numpy() - test_predictions.cpu().numpy()
    plt.hist(errors.ravel(), bins=50, color='blue', alpha=0.7)
    plt.title("Error Distribution")
    plt.xlabel("Error Value")
    plt.ylabel("Frequency")
    plt.show()

    # Plot gradient difference
    gradients_truth = np.gradient(test_ground_truth.cpu().numpy(), axis=(-2, -1))
    gradients_pred = np.gradient(test_predictions.cpu().numpy(), axis=(-2, -1))
    grad_diff_x = np.abs(gradients_truth[0] - gradients_pred[0])
    grad_diff_y = np.abs(gradients_truth[1] - gradients_pred[1])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.mean(grad_diff_x, axis=0), cmap='coolwarm')
    plt.title("Gradient Difference X")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.mean(grad_diff_y, axis=0), cmap='coolwarm')
    plt.title("Gradient Difference Y")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
