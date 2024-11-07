import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DeepUQSurrogate(nn.Module):
    def __init__(self, D, L, d, output_size=1, baselr=1e-3, lmbda=1e-6):
        """
        D -> Input dimensionality.
        L -> Number of layers in encoding section.
        d -> Size of the encoding layer.
        output_size -> Size of the output layer (default: 1)
        baselr -> Base learning rate (default: 1e-3)
        lmbda -> Regularization constant. (default: 1e-6)
        """
        super(DeepUQSurrogate, self).__init__()

        self.D = int(D)
        self.L = int(L)
        self.d = int(d)
        # self.output_size = int(D) ne fonctionne pas
        self.output_size = output_size  # Added output_size parameter
        self.baselr = baselr
        self.lmbda = lmbda
        
        # Calculate dimensions for each layer
        self.rho = (1. / self.L) * np.log(self.d / (self.D * 1.))
        self.ds = [self.D] + [int(np.ceil(self.D * np.exp(self.rho * i))) for i in range(1, self.L + 1)] + [300 * self.d, self.output_size]
        
        # Initialize layers
        layers = []
        for i in range(1, len(self.ds)):
            hin = self.ds[i - 1]
            hout = self.ds[i]
            layers.append(nn.Linear(hin, hout))
            if hout != self.output_size:  # Apply ReLU only for non-output layers
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

# Set up model, optimizer, and loss function
def create_model(D, L, d, output_size=1, baselr=1e-3, lmbda=1e-6):
    model = DeepUQSurrogate(D, L, d, output_size, baselr, lmbda)
    optimizer = optim.Adam(model.parameters(), lr=baselr)
    mse_loss = nn.MSELoss()
    return model, optimizer, mse_loss

# Training function
def train_step(model, optimizer, criterion, x, y_true):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()
    return loss.item()

# Example usage
D, L, d, output_size = 10, 3, 50, 1024  # Define dimensions, ensure output_size matches the expected output
x = torch.randn((32, D))  # Example input data
y_true = torch.randn((32, output_size))  # Example output data with correct shape

model, optimizer, criterion = create_model(D, L, d, output_size)
loss = train_step(model, optimizer, criterion, x, y_true)
print("Loss:", loss)

# Prediction
with torch.no_grad():
    y_pred = model.predict(x)
    print("Predictions:", y_pred)
