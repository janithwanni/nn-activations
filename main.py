import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

DEVICE = "mps"

# Generate a checkerboard dataset
def generate_checkerboard(n_samples, noise=0.1, random_state=None):
    np.random.seed(random_state)
    lspace = np.linspace(0, 1, n_samples)
    X = np.array(list(product(lspace, lspace)))
    # Fine-grained label assignment
    grid_size = 5  # Number of subdivisions in each dimension
    x_bins = np.floor((X[:, 0] + 1) * grid_size / 2).astype(int)  # Map x-coordinates to grid indices
    y_bins = np.floor((X[:, 1] + 1) * grid_size / 2).astype(int)  # Map y-coordinates to grid indices

    # Assign labels based on a checkerboard pattern
    y = ((x_bins + y_bins) % 2).astype(int)
    # X += noise * np.random.randn(n_samples, 2)
    return X, y

# Get column indices of non-zero values for each row
def get_nonzero_indices_per_row(tensor, level):
    non_zero_indices = torch.nonzero(tensor, as_tuple=False)  # Get indices of non-zero values
    row_indices = non_zero_indices[:, 0]  # Extract row indices
    col_indices = non_zero_indices[:, 1]  # Extract column indices
    
    # Group column indices by row
    result = {row.item(): [] for row in torch.unique(row_indices)}
    for row, col in zip(row_indices, col_indices):
        result[row.item()].append(f"{level},{col.item()}")
    
    return result

def get_circuits(groupings):
    circuits = {i: [] for k in groupings.keys() for i in groupings[k].keys()}
    for k in groupings.keys():
        for i in groupings[k].keys():
            circuits[i] += groupings[k][i]
    return circuits

# Define the neural network
class InterpretableNN(nn.Module):
    def __init__(self):
        super(InterpretableNN, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(4, 4)
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.activations = {}
        self.groupings = {}
        self.circuits = {}

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

# Generate grid data
X, y = generate_checkerboard(100)
X_tensor = torch.tensor(X, dtype=torch.float32, device = DEVICE)
y_tensor = torch.tensor(y, dtype=torch.float32, device = DEVICE).unsqueeze(1)
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
model = InterpretableNN()
model.to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Fit model
circuit_history = []
groupings = {}
circuits = {}
for epoch in range(100):
    for batch_X, batch_y in dataloader:
        predictions = model(batch_X)
        # circuit_history.append(circuits)
        loss = criterion(predictions, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# model.eval()
preds = model(X_tensor)
groupings["h1"] = get_nonzero_indices_per_row(model.activations["h1"], "h1")
groupings["h2"] = get_nonzero_indices_per_row(model.activations["h2"], "h2")
circuits = get_circuits(groupings)

# Apply model to each ij group and obtain activations for each ij group

# Count occurrences

# Visualize occurences

from collections import Counter

circ_count = Counter([frozenset(v) for k, v in circuits.items()])
circ_count.most_common()
