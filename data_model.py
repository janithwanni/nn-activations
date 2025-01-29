import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

DEVICE = "mps"

class InterpretableNN(nn.Module):
    def __init__(self, layer_sizes):
        super(InterpretableNN, self).__init__()
        self.layer1 = nn.Linear(2, layer_sizes[0])
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(layer_sizes[1], 1)
        self.sigmoid = nn.Sigmoid()
        self.activations = {}

    def forward(self, x):
        self.activations = {}
        x = self.layer1(x)        
        x = self.activation1(x)
        self.activations["h1"] = x
        
        x = self.layer2(x)
        x = self.activation2(x)
        self.activations["h2"] = x
        
        x = self.output(x)
        x = self.sigmoid(x)
        return x

def run_model(X, y, epochs = 100, layer_sizes = [4, 4], torch_seed = 123):
    torch.manual_seed(torch_seed)
    X_tensor = torch.tensor(X, dtype=torch.float32, device = DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32, device = DEVICE).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # Define model
    model = InterpretableNN(layer_sizes=layer_sizes)
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            # circuit_history.append(circuits)
            loss = criterion(predictions, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def model_boundary(model, n_samples=80, lower=-10, upper=10):
    x = np.linspace(lower, upper, n_samples)
    X = np.array(list(product(x,x)))
    X_tensor = torch.tensor(X, dtype=torch.float32, device = DEVICE)
    preds = model(X_tensor)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:,0], X[:,1], c=(preds.cpu().detach().numpy() > 0.5).astype(int), alpha=0.6, cmap=plt.cm.RdYlBu, s=8)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()