import torch
from data_utils import *
from model_utils import *
from neuron_plot import *
from concurrent.futures import ProcessPoolExecutor
import secrets
import matplotlib.pyplot as plt

# ===== 

n_samples = 6000
displ = 0
freq = 0.5
amp = 2
margin = 1
inmargin_sample = 1
outmargin_sample = 1
angle = 45
overlap = 0

data_dict = generate_data(
            n_samples, 
            {"displ":displ,"freq":freq,"amp":amp},
            {"margin":margin, "margin_sample": inmargin_sample, "outmargin_sample": outmargin_sample},
            {"x_upper":10,"x_lower":-10, "y_upper":10,"y_lower":-10},
            angle,
            overlap
        )

# ======

def test_seed(seed, data_dict=data_dict):
    # fixed data
    # start model with different seed

    epochs = 100
    layer_sizes = [5]

    data = np.vstack([data_dict["x"],data_dict["y"]]).transpose()
    y = np.array([1.0 if v == "A" else 0.0 for v in data_dict["class"]])
    model = run_model(
        data,
        y,
        epochs=epochs,
        layer_sizes = layer_sizes,
        torch_seed=seed
    )
    torch.save(model, f"checkpoints/model_{seed}.pth")

    visobj = ModelVis(model, -10, 10, 10000)
    plt.ioff()
    fig = visobj.model_boundary()
    fig.savefig(f"images/model_boundary_{seed}.png", dpi = 300)
    # obtain model boundary as an image

# ===== 

if __name__ == "__main__":
    seeds_to_try = [secrets.randbelow(99999) for i in range(600)]
    with ProcessPoolExecutor() as executor:
        executor.map(test_seed, seeds_to_try)
