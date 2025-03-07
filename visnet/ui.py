from shiny import ui

data_opts = [
    ui.input_slider(
        "n_samples",
        label="Number of samples",
        min=100,max=10000,step=100,value=6000
    ),
    ui.input_slider(
        "displ",
        label="Displacement",
        min=0,max=5,step=0.01,value=0
    ),
    ui.input_slider(
        "freq",
        label="Frequency",
        min=0,max=5,step=0.1,value=0.5
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
        min=0,max=90,step=5,value=45
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
]

model_opts = [
    ui.input_slider(
        "epochs",
        label = "Epochs",
        min = 10, max = 200, step = 10, value = 60
    ),
    ui.input_text(
        "num_neurons",
        label = "Number of neurons",
        value = "5"
    ),
    ui.input_numeric(
        "seed",
        label = "Seed",
        value = 12345
    )
]