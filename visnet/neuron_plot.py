import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import torch
from model_utils import DEVICE

class ModelVis():
    def __init__(self, model, lower, upper, n_samples):
        grid_size = round(np.sqrt(n_samples))
        x = np.linspace(lower, upper, grid_size)
        D = np.array(list(product(x, x)))
        D = D[D[:,1].argsort()]
        D_tensor = torch.tensor(D, dtype=torch.float32, device = DEVICE)
        Y = model(D_tensor)

        self.lower = lower 
        self.upper = upper
        self.D_arr = D
        self.D_tensor = D_tensor
        self.activations = model.activations
        self.loss_history = model.loss_history
        self.metric_history = model.metric_history

        for k,v in model.activations.items():
            self.activations[k] = v.cpu().detach().numpy()
    
    def layer_plot(self, act, layer_size, title="Layer"):
        """
        act: Nx(layer_size) where N is the number of observations (ideally grid_size squared)
        """
        nrows = np.ceil(layer_size / 4).astype(int)
        ncols = 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5))
        x = 0
        for i in range(nrows):
            for j in range(ncols):
                if x > act.shape[1]:
                    break
                ax = axes[i,j] if nrows != 1 else axes[j]
                scatter = ax.scatter(
                    self.D_arr[:,0],
                    self.D_arr[:,1],
                    c=act[:, i],
                    alpha = 0.6,
                    cmap="viridis",
                    s = 8
                )
                fig.colorbar(scatter, ax=ax)
                ax.set_title(f'Weight {i+j+1}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                x += 1
        fig.suptitle(title)
        return fig    
    
    def plot_activations(self, layer_num, layer_size):
        return self.layer_plot(
            (self.activations[f"h{layer_num}"] > 0).astype(int),
            layer_size
        )

    def plot_active_areas(self, num_layers):
        act_ls = None
        for i in range(num_layers):
            act = self.activations[f"h{i+1}"]
            for j in range(act.shape[1]):
                act_ls = act_ls + (act[:,j] > 0).astype(int) if act_ls is not None else (act[:, 0] > 0).astype(int)
        fig = plt.figure()
        ax = fig.subplots()
        scatter = ax.scatter(
            self.D_arr[:,0],
            self.D_arr[:,1],
            c=act_ls,
            alpha = 0.6,
            cmap="RdYlBu",
            s = 8
        )
        fig.colorbar(scatter, ax=ax)
        return fig

    def plot_loss_history(self, num_epochs):
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(
            [i for i in range(num_epochs)],
            self.loss_history,
        )
        ax.scatter(
            [i for i in range(num_epochs)],
            self.loss_history
        )
        return fig
    
    def plot_metric_history(self, num_epochs):
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(
            [i for i in range(num_epochs)],
            self.metric_history
        )
        ax.scatter(
            [i for i in range(num_epochs)],
            self.metric_history
        )
        return fig

def layer_plot(act, layer_size, lower, upper, title, nrows = 1):
    fig, axes = plt.subplots(1, layer_size, figsize=(20, 5))
    for i in range(layer_size):
        ax = axes[i]
        act_arr = act[i, :, :]
        ax.imshow(act_arr, 
                extent=(lower, upper, lower, upper), 
                origin='upper', 
                cmap='Set1')
        ax.set_title(f'Weight {i+1}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    fig.suptitle(title)

    return fig

# apply model to the entire data space
def plot_activations(model, layers = [1,2], layer_size = 4, n_samples = 1000, lower = -10, upper = 10, DEVICE = DEVICE):
    grid_size = round(np.sqrt(n_samples))
    x = np.linspace(lower, upper, grid_size)
    D = np.array(list(product(x, x)))
    D = D[D[:,1].argsort()]
    D_tensor = torch.tensor(D, dtype=torch.float32, device = DEVICE)
    Y = model(D_tensor)

    plots = []
    h1_act = model.activations["h1"].cpu().detach().numpy()
    h2_act = model.activations["h2"].cpu().detach().numpy()
    h1_act = np.vstack([h1_act[:,i].reshape(grid_size,grid_size) for i in range(layer_size)]).reshape((layer_size, grid_size, grid_size))
    h2_act = np.vstack([h2_act[:,i].reshape(grid_size,grid_size) for i in range(layer_size)]).reshape((layer_size, grid_size, grid_size))
    if(1 in layers):
        plots.append(layer_plot((h1_act >= 0).astype(int), layer_size=layer_size, lower=lower, upper=upper, title="Layer 1"))
    if(2 in layers):
        plots.append(layer_plot((h2_act >= 0).astype(int), layer_size=layer_size, lower=lower, upper=upper, title="Layer 2"))
    return plots

def plot_active_areas(model, layer_size = 4, n_samples = 1000, lower = -10, upper = 10, DEVICE = DEVICE):
    grid_size = round(np.sqrt(n_samples))
    x = np.linspace(lower, upper, grid_size)
    D = np.array(list(product(x, x)))
    D = D[D[:,1].argsort()]
    D_tensor = torch.tensor(D, dtype=torch.float32, device = DEVICE)
    Y = model(D_tensor)

    h1_act = model.activations["h1"].cpu().detach().numpy()
    h2_act = model.activations["h2"].cpu().detach().numpy()
    h1_act = np.vstack([h1_act[:,i].reshape(grid_size,grid_size) for i in range(layer_size)]).reshape((layer_size, grid_size, grid_size))
    h2_act = np.vstack([h2_act[:,i].reshape(grid_size,grid_size) for i in range(layer_size)]).reshape((layer_size, grid_size, grid_size))
    h1_act, h2_act = (h1_act >= 0).astype(int), (h2_act >= 0).astype(int)
    
    summed_activations = (h1_act.sum(axis=0) + h2_act.sum(axis=0))  # sum across grids (layer_size, grid_size, grid_size)

    # Plot each layer as a 2D heatmap
    fig = plt.figure()
    ax = fig.subplots()
    
    im = ax.imshow(summed_activations, extent=(lower, upper, lower, upper), 
                origin='upper', 
                cmap='Set3')
    ax.set_title("Most active areas")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig