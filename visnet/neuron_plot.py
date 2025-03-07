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
        self.Y = model(D_tensor)

        self.lower = lower 
        self.upper = upper
        self.D_arr = D
        self.D_tensor = D_tensor
        self.activations = model.activations
        self.loss_history = model.loss_history
        self.metric_history = model.metric_history
        self.results = model.results
        
        self.NCOLS = 5

        for k,v in model.activations.items():
            self.activations[k] = v.cpu().detach().numpy()
    
    def layer_plot(self, act, layer_size, title="Layer"):
        """
        act: Nx(layer_size) where N is the number of observations (ideally grid_size squared)
        """
        ncols = self.NCOLS
        nrows = np.ceil(layer_size / ncols).astype(int)

        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5))
        
        for i in range(nrows):
            for j in range(ncols):
                if ((i*ncols)+j) >= act.shape[1]:
                    break
                ax = axes[i,j] if nrows != 1 else axes[j]
                scatter = ax.scatter(
                    self.D_arr[:,0],
                    self.D_arr[:,1],
                    c=np.round(act[:, ((i*ncols)+j)]),
                    alpha = 0.6,
                    cmap="Paired",
                    s = 8
                )
                handles, _ = scatter.legend_elements(prop="colors")
                legend1 = ax.legend(handles, ["Not Active", "Active"], loc="lower right", fontsize = "xx-small")
                ax.add_artist(legend1)
                ax.set_title(f'Weight {(i*ncols)+j+1}')
                self.set_labels(ax)
        fig.suptitle(title)
        return fig    
    
    def plot_activations(self, layer_num, layer_size):
        return self.layer_plot(
            (self.activations[f"h{layer_num}"] > 0).astype(int),
            layer_size
        )
    
    # TODO: This is copied and should be refactored better
    def plot_results(self, res, layer_size):

        ncols = self.NCOLS
        nrows = np.ceil(layer_size / ncols).astype(int)

        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5))
        
        for i in range(nrows):
            for j in range(ncols):
                if ((i*ncols)+j) >= res.shape[1]:
                    break
                ax = axes[i,j] if nrows != 1 else axes[j]
                scatter = ax.scatter(
                    self.D_arr[:,0],
                    self.D_arr[:,1],
                    c=np.round(res[:, ((i*ncols)+j)]),
                    alpha = 0.6,
                    cmap="RdYlBu",
                    s = 8,
                    vmin = -2, vmax = 2
                )
                fig.colorbar(scatter, ax=ax)
                ax.set_title(f'Weight {(i*ncols)+j+1}')
                self.set_labels(ax)
        return fig   

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
            cmap="YlGn",
            s = 8
        )
        fig.colorbar(scatter, ax=ax)
        ax.set_title("No. of times an area was active")
        self.set_labels(ax)

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
            self.loss_history,
            s = 0.1
        )
        ax.set_title(f"Last loss {round(self.loss_history[-1],2)}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
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
            self.metric_history,
            s=1
        )
        ax.set_title(f"Last Accuracy {round(self.metric_history[-1],2)}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        return fig
    
    def make_reg_line(self, weights, biases, yc = 0):
        """
        weights: A array of layer_size x 2
        biases: A array of layer_size

        Returns: A array of layer_size x 2 indicating intercept and slope
        """
        n_neurons  = weights.shape[0]
        result = np.empty((n_neurons, 2))
        for i in range(n_neurons):
            result[i, 0] = -(weights[i,0] / weights[i,1])
            result[i, 1] = -(biases[i] - yc) / weights[i, 1] # fixing y as 0
        return result

    def plot_reg_line(self, weights, biases):
        """
        weights: A array of layer_size x 2
        biases: A array of layer_size

        Returns: A list of plots of regression line
        """
        n_neurons = weights.shape[0]
        reg_line = self.make_reg_line(weights, biases)

        x = np.linspace(-10,10,1000)

        fig = plt.figure()
        ax = fig.subplots()
        ax.set_xlim([-10,10])
        ax.set_ylim([-10,10])
        self.set_labels(ax)
        for i in range(n_neurons):
            y = reg_line[i, 0] * x + reg_line[i, 1]
            color = [c / 255 for c in plt.cm.get_cmap("Dark2")(i, bytes=True)]
            ax.plot(x,y,c=color, label = f"w{i+1}")
        if n_neurons <= 8:
            fig.legend(loc = "lower right", fontsize="xx-small")
        ax.set_title("Regression from layer 1 weights")
        return fig
    
    def model_boundary(self):
        plt.close()
        fig = plt.figure()
        ax = fig.subplots()
        scatter = ax.scatter(
            self.D_arr[:,0],
            self.D_arr[:,1], 
            c=self.Y.cpu().detach().numpy(), 
            alpha=0.6, 
            cmap="RdYlBu",
            s=8
        )
        fig.colorbar(scatter, ax=ax)
        ax.set_title("Model probabilities")
        self.set_labels(ax)
        return fig

    def set_labels(self, ax):
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
