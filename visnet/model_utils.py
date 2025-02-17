import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

DEVICE = "cpu"
torch.set_default_device('cpu')

class InterpretableNN(nn.Module):
    def __init__(self, layer_sizes):
        super(InterpretableNN, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.layers = [[None] for i in range(len(layer_sizes))]

        self.layers[0] = [
            nn.Linear(2, layer_sizes[0]),
            nn.ReLU()
        ]

        for i in range(1, len(layer_sizes)):
            self.layers[i] = [
                nn.Linear(layer_sizes[(i-1)], layer_sizes[i]),
                nn.ReLU()
            ]

        self.output = nn.Linear(layer_sizes[-1], 1)
        self.activations = {}
        self.loss_history = []
        self.metric_history = []

    def forward(self, x):
        self.activations = {}
        for i, nl in enumerate(self.layers):
            for nly in nl:
                x = nly(x)
            self.activations[f"h{(i+1)}"] = x
        
        x = self.output(x)
        x = nn.Sigmoid()(x)
        return x

def run_model(X, y, epochs = 100, layer_sizes = [4, 4], torch_seed = 123):
    torch.manual_seed(torch_seed)
    X_tensor = torch.tensor(X, dtype=torch.float32, device = DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32, device = DEVICE).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=round(np.sqrt(X_tensor.shape[0])),
        shuffle=True
    )

    # Define model
    model = InterpretableNN(layer_sizes=layer_sizes)
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    model.loss_history = []

    for epoch in range(epochs):
        correct = 0
        loss_epoch = 0
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            # circuit_history.append(circuits)
            loss = criterion(predictions, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct_epoch = ((predictions > 0.5).int() == batch_y).int().sum().item()
            correct += correct_epoch
            loss_epoch += (loss.item() * batch_y.shape[0])
        correct /= len(dataloader.dataset)
        model.metric_history.append(correct * 100)
        model.loss_history.append(loss_epoch / len(dataloader.dataset))
    return model

def model_boundary(model, n_samples=75, lower=-10, upper=10):
    x = np.linspace(lower, upper, n_samples)
    X = np.array(list(product(x,x)))
    X_tensor = torch.tensor(X, dtype=torch.float32, device = DEVICE)
    preds = model(X_tensor)
    
    fig = plt.figure()
    ax = fig.subplots()
    scatter = ax.scatter(
        X[:,0],
        X[:,1], 
        c=preds.cpu().detach().numpy(), 
        alpha=0.6, 
        cmap=plt.cm.RdYlBu,
        s=8
    )
    fig.colorbar(scatter, ax=ax)
    return fig
