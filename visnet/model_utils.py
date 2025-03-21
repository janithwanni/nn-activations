import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

DEVICE = "cpu"
torch.set_default_device(DEVICE)

class InterpretableNN(nn.Module):
    def __init__(self, layer_sizes):
        super(InterpretableNN, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(2, layer_sizes[0]))
        self.layers.append(nn.ReLU())

        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[(i-1)], layer_sizes[i]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(layer_sizes[-1], 1))
        self.layers.append(nn.Sigmoid())

        self.activations = {}
        self.results = {}
        self.loss_history = []
        self.metric_history = []

    def forward(self, x):
        self.activations = {}
        for i, l in enumerate(self.layers):
            x = l(x)

            if isinstance(l, nn.Linear):
                label = "Linear"
            if isinstance(l, nn.ReLU):
                label = "ReLu"
            if isinstance(l, nn.Sigmoid):
                label = "Sigmoid"
            
            if i == (len(self.layers) - 3):
                a = x.detach().clone()
                weight_layer = list(self.layers[i+1].parameters())
                
                for i in range(x.shape[1]):
                    a[:, i] = a[:, i] * weight_layer[0][0, i] + weight_layer[1][0]
                self.results[f"interim-{i}"] = a
            self.results[f"{label}-{i}"] = x
            if i % 2 != 1: # odd number of layers are activations
                self.activations[f"h{int((i+2)/2)}"] = x
        return x

def run_model(X, y, epochs = 100, layer_sizes = [4, 4], torch_seed = 123):
    torch.manual_seed(torch_seed)
    X_tensor = torch.tensor(X, dtype=torch.float32, device = DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32, device = DEVICE).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=round(np.sqrt(X_tensor.shape[0])),
        shuffle=True,
        generator = torch.Generator(device=DEVICE).manual_seed(torch_seed)
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

def fit_single_model(seed, n_neurons, train_df):
    epochs = 100
    layer_sizes = [n_neurons]

    data = train_df.loc[:, ["x", "y"]].values
    y = np.array([1.0 if v == "A" else 0.0 for v in train_df.loc[:, "class"].values.tolist()])
    model = run_model(
        data,
        y,
        epochs=epochs,
        layer_sizes = layer_sizes,
        torch_seed=seed
    )

    return model

def predict(model, df):
    D_tensor = torch.tensor(df.loc[:, ["x", "y"]].values, dtype=torch.float32, device = DEVICE)
    preds = (model(D_tensor) > 0.5).int().cpu().detach().numpy().tolist()
    # preds = ["A" if v == 1.0 else "B" for v in preds]
    return preds 

def evaluate(model, df):
  # get predictions 
  preds = predict(model, df)
  ground = [1 if v == "A" else 0 for v in df.loc[:, "class"].values.tolist()]

  # get f1 score 
  f_val = f1_score(preds,ground)
  # get accuracy 
  accuracy = accuracy_score(preds, ground)
  return {"f": f_val, "acc": accuracy}
