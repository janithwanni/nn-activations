import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import torch
from model_utils import DEVICE

def layer_plot(act, layer_size, lower, upper, title, nrows = 1):
    fig, axes = plt.subplots(1, layer_size, figsize=(20, 5))
    for i in range(layer_size):
        ax = axes[i]
        ax.imshow(act[:, :, i].cpu().detach().numpy(), 
                extent=(lower, upper, lower, upper), 
                origin='lower', 
                cmap='viridis')
        ax.set_title(f'Weight {i+1}')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
    fig.suptitle(title)

    return fig

# apply model to the entire data space
def plot_activations(model, layers = [1,2], layer_size = 4, n_samples = 100, lower = -10, upper = 10, DEVICE = DEVICE):
    grid_size = round(np.sqrt(n_samples))
    x = np.linspace(lower, upper, grid_size)
    D = np.array(list(product(x, x)))
    D_tensor = torch.tensor(D, dtype=torch.float32, device = DEVICE)
    Y = model(D_tensor)

    plots = []
    if(1 in layers):
        h1_act = model.activations["h1"].reshape((grid_size, grid_size, layer_size))
        plots.append(layer_plot(h1_act, layer_size=layer_size, lower=lower, upper=upper, title="Layer 1"))
    if(2 in layers):
        h2_act = model.activations["h2"].reshape((grid_size, grid_size, layer_size))
        plots.append(layer_plot(h2_act, layer_size=layer_size, lower=lower, upper=upper, title="Layer 2"))
    print(len(plots))
    return plots

def plot_active_areas(model, layer_size = 4, n_samples = 100, lower = -10, upper = 10, DEVICE = DEVICE):
    grid_size = round(np.sqrt(n_samples))
    x = np.linspace(lower, upper, grid_size)
    D = np.array(list(product(x, x)))
    D_tensor = torch.tensor(D, dtype=torch.float32, device = DEVICE)
    Y = model(D_tensor)

    h1_act = model.activations["h1"].reshape((grid_size, grid_size, layer_size))
    h2_act = model.activations["h2"].reshape((grid_size, grid_size, layer_size))

    summed_activations = (h1_act + h2_act).sum(dim=2)  # Element-wise sum, retains shape (grid_size, grid_size, layer_size)

    # Plot each layer as a 2D heatmap
    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots()
    ax.imshow(summed_activations.cpu().detach().numpy(), extent=(lower, upper, lower, upper), 
                origin='lower', 
                cmap='viridis')
    ax.set_title("Most active areas")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return fig