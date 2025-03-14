from shiny import *
import pandas as pd
import glob
from model_utils import *
from neuron_plot import *
import plotnine as pn 

app_ui = ui.page_fluid(
    ui.layout_columns(
        ui.div(
            ui.output_data_frame("leaderboard"),
            ui.output_plot("leader_plot")
        ),
        ui.div(
            ui.input_selectize(
                "plot_view",
                "Choose view",
                {"bound": "Model boundary", "misclass": "Misclassified in testing"}
            ),
            ui.output_plot("boundary"),
        ),
        col_widths=[6,6]
    )
)

def server(input, output, session):
    train_df = pd.read_csv("train_df.csv")
    test_df = pd.read_csv("test_df.csv")
    true_df = pd.read_csv("df.csv")
    csvs = glob.glob("res_df*.csv")
    df = pd.concat([pd.read_csv(v) for v in csvs])
    df = df.loc[df.epoch != 100, :]

    @render.data_frame
    def leaderboard():
        return render.DataGrid(df, selection_mode= "row", width = "50vw")

    @render.plot
    def leader_plot():
        pt = (
            pn.ggplot(df, pn.aes("f1", "acc")) +
            pn.geom_point() +
            pn.facet_wrap("neuron") +
            pn.theme_minimal()
        )
        return pt
   
    @reactive.calc 
    def model():
        selected_row = df.iloc[list(input.leaderboard_cell_selection()["rows"]), :]
        req(selected_row.shape[0] != 0)
        model = fit_single_model(
            int(selected_row.seed.values[0]), 
            int(selected_row.neuron.values[0]), 
            train_df
        )
        return model

    @render.plot
    def boundary():
        preds = predict(model(), test_df)
        preds = ["A" if p[0] == 1 else "B" for p in preds]
        targets = test_df.loc[:, "class"].values.tolist()
        misses = [preds[i] != targets[i] for i in range(test_df.shape[0])]
        model_vis = ModelVis(
            model(),
            n_samples = 10000
        )
        if input.plot_view() == "misclass":
            fig_data = pd.DataFrame({
                "x": test_df.loc[:, "x"].values,
                "y": test_df.loc[:, "y"].values,
                "misses": misses
            })
            fig = (
                pn.ggplot(fig_data, pn.aes("x","y",color = "factor(misses)"))+
                pn.geom_point()+
                pn.theme_minimal()
            )
        else: 
            fig = model_vis.model_boundary()

        return fig

app = App(app_ui, server)
