from shiny import App, render, ui
import numpy as np
import matplotlib.pyplot as plt
from shiny import ui, App, render, reactive, req
from data_utils import *
from model_utils import *
from neuron_plot import *
from ui import data_opts, model_opts

PLOT_DIM = "40vh"

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
    ui.div(
        ui.output_plot(
            "loss_history",
            height = "20vh", width = "30vw"
        ),
        ui.output_plot(
            "metric_history",
            height = "20vh", width = "30vw"
        ),
        ui.row(    
            ui.div(ui.output_text("avg_loss"), fill = False),
            ui.div(ui.output_text("avg_metric"), fill = False)
        )
    ),
    ui.div(
        ui.output_plot(
            "data",
            height = PLOT_DIM, width = PLOT_DIM
        ),
        ui.output_plot(
            "model_bound",
            height = PLOT_DIM, width = PLOT_DIM
        ),
        ui.output_plot(
            "active_areas",
            height = PLOT_DIM, width = PLOT_DIM
        )
    ),
    ui.div(
        ui.input_select("layer_select", "Select layer", []),
        ui.output_plot(
            "l1_activations",
            height = PLOT_DIM, width = "80vw"
        ),
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
        return fig
    
    @render.plot
    def model_bound():
        fig = model_boundary(model())
        return fig

    @render.plot
    def l1_activations():
        l = input.layer_select()
        req(l)
        return visobj().plot_activations(
            int(l)+1, layers()[int(l)]
        )

    @render.plot
    def l2_activations():
        return None
    
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

    @render.text
    def avg_loss():
        return f"avg loss {round(model().loss_history[-1], 2)}"
    
    @render.text
    def avg_metric():
        return f"avg acc {round(model().metric_history[-1], 2)}"
    

app = App(app_ui, server, debug=False)
