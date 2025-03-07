from shiny import App, render, ui
import numpy as np
import matplotlib.pyplot as plt
from shiny import ui, App, render, reactive, req
from data_utils import *
from model_utils import *
from neuron_plot import *
from ui import data_opts, model_opts
from pathlib import Path

PLOT_DIM = "40vh"
PLOT_WIDTH = "60vw"

app_ui = ui.page_sidebar(
    ui.sidebar(
        # TODO: Add an accordion to group the sliders
        ui.accordion(
            ui.accordion_panel(
                "Data generation options",
                *data_opts,
            ),
            ui.accordion_panel(
                "Model building options",
                *model_opts
            ),
            open = "Model building options"
        )
    ),
    ui.head_content(ui.include_css(Path(__file__).parent / "main.css")),
    ui.accordion(
        ui.accordion_panel(
            "History",
            ui.div(
                ui.output_plot(
                    "loss_history",
                    height = "20vh", width = "30vw"
                ),
                ui.output_plot(
                    "metric_history",
                    height = "20vh", width = "30vw"
                ),
                class_ = "history_plots"
            ),
            {"data-id" : "main-plots-container"}
        ),
        open = False
    ),
    ui.accordion(
        ui.accordion_panel(
            "Model diagnostics",
            ui.div(
                ui.output_plot(
                    "data",
                    height = PLOT_DIM, width = PLOT_WIDTH
                ),
                ui.output_plot(
                    "model_bound",
                    height = PLOT_DIM, width = PLOT_WIDTH
                ),
                ui.output_plot(
                    "active_areas",
                    height = PLOT_DIM, width = PLOT_WIDTH
                ),
                ui.output_plot(
                    "weight_lines",
                    height = PLOT_DIM, width = PLOT_WIDTH
                ),
                class_ = "main_plots"
            )
        )
    ),
    ui.div(
        ui.input_select("layer_select", "Select layer", []),
        ui.output_ui("layer_activations"),
        class_ = "layer_act_plots"
    ),
    ui.div(
        ui.input_select("result_select", "Select intermediate result", []),
        ui.output_ui("result_plots_container"),
        class_ = "result_plot_outputs"
    ),
    title = "VisNet",
    fillable = True, fillable_mobile = True
)


def server(input, output, session):
    @reactive.calc
    def gen_data():
        # req(
        #     input.n_samples(),
        #     input.displ(), input.freq(), input.amp(),
        #     input.angle(), input.margin(), input.outmargin_sample(), input.inmargin_sample(),
        #     input.overlap()
        # )
        n_samples = input.n_samples()
        displ = input.displ()
        freq = input.freq()
        amp = input.amp()
        angle = input.angle()
        margin = input.margin()
        outmargin_sample = input.outmargin_sample()
        inmargin_sample = input.inmargin_sample()
        overlap = input.overlap()

        return generate_data(
            n_samples, 
            {"displ":displ,"freq":freq,"amp":amp},
            {"margin":margin, "margin_sample": inmargin_sample, "outmargin_sample": outmargin_sample},
            {"x_upper":10,"x_lower":-10, "y_upper":10,"y_lower":-10},
            angle,
            overlap
        )
    
    @reactive.calc
    def layers():
        neurons = input.num_neurons()
        layer_sizes = [int(x) for x in neurons.split(",") if x != '']
        return layer_sizes
    
    @reactive.effect
    def update_layer_selector():
        ui.update_select(
            "layer_select",
            choices = {
                i: f"Layer {i+1}" for i in range(len(layers()))
            }
        )
        ui.update_select(
            "result_select",
            choices = {
                val: f"{val}" for i, val in enumerate(model().results.keys())
            }
        )
    
    @reactive.calc
    def model():
        d = gen_data()
        epochs = input.epochs()
        layer_sizes = layers()
        data = np.vstack([d["x"],d["y"]]).transpose()
        y = np.array([1.0 if v == "A" else 0.0 for v in d["class"]])
        model = run_model(
            data,
            y,
            epochs=epochs,
            layer_sizes = layer_sizes
        )
        # torch.save(model.state_dict(), "visnet/model_chkpoint.pth")
        return model

    @reactive.calc
    def visobj():
        visobj = ModelVis(model(), -10, 10, 10000)
        return visobj

    @render.plot
    def data():
        D = gen_data()
        fig = plt.figure()
        ax = fig.subplots()
        ax.scatter(x=D["x"],y=D["y"],c=np.where(D["class"] == "A", 1,0))
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_title("Dataset")
        return fig
    
    @render.plot
    def model_bound():
        fig = visobj().model_boundary()
        return fig
    
    @render.ui
    def layer_activations():
        req(input.layer_select())
        l = input.layer_select()
        layer_size = layers()[int(l)]
        ncols = visobj().NCOLS
        nrows = np.ceil(layer_size / ncols).astype(int)
        height = nrows * 30
        return ui.output_plot(
            "l1_activations",
            height = f"{height}vh", width = "80vw"
        )
    @render.plot
    def l1_activations():
        l = input.layer_select()
        req(l)
        return visobj().plot_activations(
            int(l)+1, layers()[int(l)]
        )
    @render.ui
    def result_plots_container():
        req(input.result_select())
        layer_key = input.result_select()
        # print("=======")
        # print(layer_key)
        # print(model().results[layer_key].shape[1])
        # print("========")
        ncols = visobj().NCOLS
        layer_size = model().results[layer_key].shape[1]
        nrows = np.ceil(layer_size / ncols).astype(int)
        height = nrows * 30
        return ui.output_plot(
            "result_outputs",
            height = f"{height}vh", width = "80vw"
        )
    @render.plot
    def result_outputs():
        l = input.result_select()
        req(l)
        return visobj().plot_results(
            model().results[l].cpu().detach().numpy(),
            layer_size = model().results[l].shape[1]
        )
    @render.plot
    def weight_lines():
        weights = model().state_dict()["layers.0.weight"].numpy()
        biases = model().state_dict()["layers.0.bias"].numpy()
        return visobj().plot_reg_line(weights, biases)
    
    @render.plot
    def active_areas():
        return visobj().plot_active_areas(len(layers()))

    @render.plot
    def loss_history():
        epochs = input.epochs()
        return visobj().plot_loss_history(epochs)

    @render.plot
    def metric_history():
        epochs = input.epochs()
        return visobj().plot_metric_history(epochs)
    

app = App(app_ui, server, debug=False)
