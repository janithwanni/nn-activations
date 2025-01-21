from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Generate a checkerboard dataset
def generate_checkerboard(n_samples, noise=0.1, random_state=None):
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 2) * 2 - 1
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += noise * np.random.randn(n_samples, 2)
    return X, y

# Generate dataset
n_samples = 1000
noise = 0.1
random_state = 42
X, y = generate_checkerboard(n_samples=n_samples, noise=noise, random_state=random_state)

# Define the neural network
class InterpretableNN(nn.Module):
    def __init__(self):
        super(InterpretableNN, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 16)
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.activations = {}

    def forward(self, x):
        self.activations['input'] = x
        x = self.layer1(x)
        self.activations['layer1'] = x
        x = self.activation1(x)
        self.activations['activation1'] = x
        x = self.layer2(x)
        self.activations['layer2'] = x
        x = self.activation2(x)
        self.activations['activation2'] = x
        x = self.output(x)
        self.activations['output'] = x
        x = self.sigmoid(x)
        self.activations['sigmoid'] = x
        return x

# Train the model
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
model = InterpretableNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(100):
    for batch_X, batch_y in dataloader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Shiny app
app_ui = ui.page_fluid(
    ui.h2("Interactive Neural Network Activation Explorer"),
    ui.div(
        ui.output_plot("dataset_plot", height="400px"),
        style="display:inline-block; width:48%;"
    ),
    ui.div(
        ui.output_text_verbatim("activations"),
        style="display:inline-block; width:48%; vertical-align:top;"
    ),
    ui.div(
        ui.input_select(
            "layer_select", 
            "Select Layer", 
            {"layer1": "Layer 1", "layer2": "Layer 2"}
        ),
        ui.input_numeric("neuron_select", "Neuron Index", 0, min=0, step=1),
        ui.output_plot("activation_heatmap", height="400px"),
        style="display:inline-block; width:100%; margin-top:20px;"
    )
)

def server(input, output, session):
    @reactive.Value
    def clicked_point():
        return None

    @output
    @render.plot
    def dataset_plot():
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
        ax.set_title("Click on a point to inspect activations")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        
        def onclick(event):
            if event.inaxes == ax:
                clicked_point.set((event.xdata, event.ydata))
        fig.canvas.mpl_connect('button_press_event', onclick)
        return fig

    @output
    @render.text
    def activations():
        point = clicked_point()
        if point is None:
            return "Click on a point in the dataset to view activations."
        
        data_point = torch.tensor([[point[0], point[1]]], dtype=torch.float32)
        model(data_point)
        activation_strings = [
            f"{layer}: {activation.detach().numpy().flatten()}"
            for layer, activation in model.activations.items()
        ]
        return "\n".join(activation_strings)

    @output
    @render.plot
    def activation_heatmap():
        # Get layer and neuron index from input
        selected_layer = input.layer_select()
        neuron_index = int(input.neuron_select())
        
        # Define grid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        
        # Forward pass and extract activations
        model(grid)
        activations = model.activations[selected_layer].detach().numpy()
        
        # Check neuron index validity
        if neuron_index < 0 or neuron_index >= activations.shape[1]:
            return f"Invalid neuron index. Select between 0 and {activations.shape[1] - 1}."
        
        # Plot heatmap
        activation_map = activations[:, neuron_index].reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(6, 5))
        c = ax.contourf(xx, yy, activation_map, levels=50, cmap="viridis", alpha=0.8)
        plt.colorbar(c, ax=ax)
        ax.set_title(f"Activation Heatmap for {selected_layer}, Neuron {neuron_index}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        return fig

app = App(app_ui, server)
