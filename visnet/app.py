from shiny import App, render, ui
import numpy as np
import matplotlib.pyplot as plt
from shiny import ui, App, render, reactive, req
from data_utils import *
from model_utils import *
from neuron_plot import *

PLOT_DIM = "40vh"

app_ui = ui.page_sidebar(
    ui.sidebar(
        # TODO: Add an accordion to group the sliders
        ui.input_slider(
            "n_samples",
            label="Number of samples",
            min=100,max=10000,step=100,value=5000
        ),
        ui.input_slider(
            "displ",
            label="Displacement",
            min=0,max=5,step=0.01,value=0
        ),
        ui.input_slider(
            "freq",
            label="Frequency",
            min=0,max=5,step=0.01,value=2
        ),
        ui.input_slider(
            "amp",
            label="Amplitude",
            min=0,max=5,step=0.01,value=2
        ),
        ui.input_slider(
            "margin",
            label = "Margin",
            min=0,max=5,step=0.01,value=1
        ),
        ui.input_slider(
            "angle",
            label = "Rotation angle",
            min=0,max=360,step=15,value=45
        ),
        ui.input_slider(
            "outmargin_sample",
            label = "Out of margin sampling rate",
            min=0,max=1,step=0.01,value=1
        ),
        ui.input_slider(
            "inmargin_sample",
            label = "In margin sampling rate",
            min=0,max=1,step=0.01,value=1
        ),
        ui.input_slider(
            "overlap",
            label = "Overlap",
            min=0,max=5,step=0.01,value=0
        )
    ),
    ui.layout_columns(
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
        ),
        col_widths = [4]*3
    ),
    ui.layout_columns(
        ui.output_plot(
            "l1_activations",
            height = PLOT_DIM, width = "100%"
        ),
        ui.output_plot(
            "l2_activations",
            height = PLOT_DIM, width = "100%"
        ),
        col_widths=[6]*2
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
    def model():
        d = gen_data()
        data = np.vstack([d["x"],d["y"]]).transpose()
        model = run_model(data, np.array([1.0 if v == "A" else 0.0 for v in d["class"]]), epochs=10)
        return model

    @reactive.calc
    def activations():
        return plot_activations(model(), layer_size=4)

    @render.plot
    def data():
        D = gen_data()
        fig = plt.figure()
        ax = fig.subplots()
        ax.scatter(x=D["x"],y=D["y"],c=np.where(D["class"] == "A", 1,0))
        return fig
    
    @render.plot
    def model_bound():
        fig = model_boundary(model())
        return fig

    @render.plot
    def l1_activations():
        return activations()[0]

    @render.plot
    def l2_activations():
        return activations()[1]
    
    @render.plot
    def active_areas():
        return plot_active_areas(model(), layer_size=4)

app = App(app_ui, server, debug=False)
