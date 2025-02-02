# #!/usr/bin/env python

# import numpy as np
# import os
# from examples.spde.solver import SteadyStateHeat2DSolver  # Ensure this path is correct
# import torch
# from scripts.surrogate_pytorch import DeepUQSurrogate  # Ensure surrogate_pytorch has PyTorch model
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

# # Model path and device
# model_path = os.path.join(os.path.dirname(__file__), "models", "iteration_model_4096.pth")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Model setup
# nx = ny = 32
# D = nx * ny
# L = 3  # Number of encoding layers
# d = 128  # Encoding dimension

# model = DeepUQSurrogate(D=D, L=L, d=d, output_size=D).to(device)
# model.load_state_dict(torch.load(model_path))
# model.eval()

# # Load the dataset
# # data_path = r"C:\python_workspace\3A\ProjetSOIA\deep-uq-paper-master\data\data_rbf_samples_2000_nx_32_ny_32.npz"

# # data_path = r"C:\python_workspace\3A\ProjetSOIA\deep-uq-paper-master\data\file_split\test_file\data_rbf_samples_12000_nx_32_ny_32_1999_2000.npz"

# data_path = r"C:\python_workspace\3A\ProjetSOIA\deep-uq-paper-master\data\file_4096\test_file\data_rbf_samples_40960_nx_32_ny_32_8_4096.npz"

# data = np.load(data_path)
# inputs = data['inputs'].reshape(-1, nx * ny)
# outputs = data['outputs'].reshape(-1, nx * ny)

# # Select a random test subset (20%)
# num_samples = inputs.shape[0]
# test_size = int(0.8 * num_samples)
# random_indices = np.random.choice(num_samples, test_size, replace=False)
# test_inputs = inputs[random_indices, :]
# ground_truth_output = outputs[random_indices, :]

# # Convert to tensors
# test_inputs = torch.from_numpy(test_inputs).float().to(device)
# ground_truth_output = torch.from_numpy(ground_truth_output).float().to(device)

# # Perform predictions
# with torch.no_grad():
#     test_predictions = model(test_inputs).reshape(-1, nx, ny).cpu().numpy()
#     test_ground_truth = ground_truth_output.reshape(-1, nx, ny).cpu().numpy()

# # Normalize the images to [0, 1]
# test_predictions_normalized = (test_predictions - test_predictions.min()) / (
#     test_predictions.max() - test_predictions.min()
# )
# test_ground_truth_normalized = (test_ground_truth - test_ground_truth.min()) / (
#     test_ground_truth.max() - test_ground_truth.min()
# )

# # Evaluate metrics
# mse_values = []
# ssim_values = []
# for i in range(test_predictions_normalized.shape[0]):
#     truth = test_ground_truth_normalized[i]
#     pred = test_predictions_normalized[i]
#     mse = np.mean((truth - pred) ** 2)
#     similarity = ssim(truth, pred, data_range=1.0)  # SSIM with data_range=1.0
#     mse_values.append(mse)
#     ssim_values.append(similarity)

# print(f"Average MSE: {np.mean(mse_values):.4f}")
# print(f"Average SSIM: {np.mean(ssim_values):.4f}")

# # Visualization
# indices = np.random.choice(test_predictions.shape[0], 5, replace=False)
# fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# for idx, i in enumerate(indices):
#     # Ground truth
#     axes[0, idx].imshow(test_ground_truth_normalized[i], cmap="viridis")
#     axes[0, idx].set_title(f"Ground Truth {i+1}")
#     axes[0, idx].axis("off")

#     # Prediction
#     axes[1, idx].imshow(test_predictions_normalized[i], cmap="viridis")
#     axes[1, idx].set_title(f"Prediction {i+1}")
#     axes[1, idx].axis("off")

#     # Difference (Error Map)
#     error_map = np.abs(test_ground_truth_normalized[i] - test_predictions_normalized[i])
#     axes[2, idx].imshow(error_map, cmap="inferno")
#     axes[2, idx].set_title(f"Error Map {i+1}")
#     axes[2, idx].axis("off")

#     # Gradient Magnitude of Ground Truth
#     gx, gy = np.gradient(test_ground_truth_normalized[i])
#     gradient_truth = np.sqrt(gx**2 + gy**2)
#     axes[3, idx].imshow(gradient_truth, cmap="plasma")
#     axes[3, idx].set_title(f"Gradient Truth {i+1}")
#     axes[3, idx].axis("off")

#     # Gradient Magnitude of Prediction
#     gx_pred, gy_pred = np.gradient(test_predictions_normalized[i])
#     gradient_pred = np.sqrt(gx_pred**2 + gy_pred**2)
#     axes[4, idx].imshow(gradient_pred, cmap="plasma")
#     axes[4, idx].set_title(f"Gradient Prediction {i+1}")
#     axes[4, idx].axis("off")

# plt.suptitle("Ground Truth, Predictions, Error Maps, and Gradients")
# plt.tight_layout()
# plt.show()



#!/usr/bin/env python

import numpy as np
import os
from examples.spde.solver import SteadyStateHeat2DSolver  # Ensure this path is correct
import torch
from scripts.surrogate_pytorch import DeepUQSurrogate  # Ensure surrogate_pytorch has PyTorch model
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Model path and device
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

# Select a random test subset (20%)
num_samples = inputs.shape[0]
test_size = int(0.8 * num_samples)
random_indices = np.random.choice(num_samples, test_size, replace=False)
test_inputs = inputs[random_indices, :]
ground_truth_output = outputs[random_indices, :]

# Convert to tensors
test_inputs = torch.from_numpy(test_inputs).float().to(device)
ground_truth_output = torch.from_numpy(ground_truth_output).float().to(device)

# Perform predictions
with torch.no_grad():
    test_predictions = model(test_inputs).reshape(-1, nx, ny).cpu().numpy()
    test_ground_truth = ground_truth_output.reshape(-1, nx, ny).cpu().numpy()

# Normalize the images to [0, 1]
test_predictions_normalized = (test_predictions - test_predictions.min()) / (
    test_predictions.max() - test_predictions.min()
)
test_ground_truth_normalized = (test_ground_truth - test_ground_truth.min()) / (
    test_ground_truth.max() - test_ground_truth.min()
)

# Evaluate metrics
mse_values = []
ssim_values = []
for i in range(test_predictions_normalized.shape[0]):
    truth = test_ground_truth_normalized[i]
    pred = test_predictions_normalized[i]
    mse = np.mean((truth - pred) ** 2)
    similarity = ssim(truth, pred, data_range=1.0)  # SSIM with data_range=1.0
    mse_values.append(mse)
    ssim_values.append(similarity)

print(f"Average MSE: {np.mean(mse_values):.4f}")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")

# Visualization
indices = np.random.choice(test_predictions.shape[0], 2, replace=False)

# Ground Truth and Prediction
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
for idx, i in enumerate(indices):
    # Ground truth
    im1 = axes1[0, idx].imshow(test_ground_truth_normalized[i], cmap="viridis")
    axes1[0, idx].set_title(f"Ground Truth {i+1}")
    axes1[0, idx].axis("off")

    # Prediction
    im2 = axes1[1, idx].imshow(test_predictions_normalized[i], cmap="viridis")
    axes1[1, idx].set_title(f"Prediction {i+1}")
    axes1[1, idx].axis("off")

fig1.suptitle("Ground Truth and Predictions")
plt.tight_layout()
fig1.colorbar(im1,ax=axes1, orientation='vertical', fraction=0.046, pad=0.04)
# fig1.colorbar(im1,ax=axes1.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.04)
# Use this line if there are problem with the colorbar

# Error Maps
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
for idx, i in enumerate(indices):
    error_map = np.abs(test_ground_truth_normalized[i] - test_predictions_normalized[i])
    im = axes2[idx].imshow(error_map, cmap="inferno")
    axes2[idx].set_title(f"Error Map {i+1}")
    axes2[idx].axis("off")

fig2.suptitle("Error Maps")
plt.tight_layout()
fig2.colorbar(im, ax=axes2, orientation='vertical', fraction=0.046, pad=0.04)


# Gradients
fig3, axes3 = plt.subplots(2, 2, figsize=(10, 10))
for idx, i in enumerate(indices):
    # Gradient Magnitude of Ground Truth
    gx, gy = np.gradient(test_ground_truth_normalized[i])
    gradient_truth = np.sqrt(gx**2 + gy**2)
    im = axes3[0, idx].imshow(gradient_truth, cmap="plasma")
    axes3[0, idx].set_title(f"Gradient Truth {i+1}")
    axes3[0, idx].axis("off")

    # Gradient Magnitude of Prediction
    gx_pred, gy_pred = np.gradient(test_predictions_normalized[i])
    gradient_pred = np.sqrt(gx_pred**2 + gy_pred**2)
    axes3[1, idx].imshow(gradient_pred, cmap="plasma")
    axes3[1, idx].set_title(f"Gradient Prediction {i+1}")
    axes3[1, idx].axis("off")

fig3.suptitle("Gradients of Ground Truth and Predictions")
plt.tight_layout()
fig1.colorbar(im,ax=axes3, orientation='vertical', fraction=0.046, pad=0.04)

plt.show()
