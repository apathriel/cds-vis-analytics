from pathlib import Path
from typing import Dict

from data_processing_utilities import convert_string_to_snake_case
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns


def construct_visualization_parameters(
    df, color_iteration_index, plot_title, visualization_parameters: Dict
) -> Dict:
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

    visualization_parameters.update(
        {
            "df": df,
            "plot_title": f"{plot_title} Evolution of faces in newspapers by decade",
            "line_color": colors[color_iteration_index % len(colors)],
        }
    )

    return visualization_parameters


def visualize_trend_by_time_from_df(
    df: pd.DataFrame,
    plot_title: str,
    x_axis_df_column: str,
    y_axis_df_column: str,
    save_visualization: bool,
    add_regression: bool = False,
    output_directory: Path = Path(__file__).parent,
    img_format: str = "pdf",
    line_color: str = "blue",
) -> None:
    plt.figure(figsize=(10, 6))

    if add_regression:
        plot_title = f"{plot_title} regression"
        sns.regplot(x=df[x_axis_df_column], y=df[y_axis_df_column], color=line_color)
    else:
        plt.plot(df[x_axis_df_column], df[y_axis_df_column], color=line_color)

    plt.title(plot_title.upper(), fontsize=16, color="black", family="Verdana")
    plt.xlabel(x_axis_df_column.upper(), fontsize=12, color="black", family="Verdana")
    plt.ylabel(y_axis_df_column.upper(), fontsize=12, color="black", family="Verdana")
    plt.grid(True)

    if save_visualization:
        output_directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            f"{output_directory}/{convert_string_to_snake_case(plot_title)}.{img_format}"
        )
    else:
        plt.show()


def interactive_visualization_trend_by_time_from_df(
    df: pd.DataFrame,
    plot_title: str,
    x_axis_df_column: str,
    y_axis_df_column: str,
    save_visualization: bool,
    output_directory: Path = Path(__file__).parent,
    img_format: str = "pdf",
    line_color: str = "blue",
) -> None:
    fig = px.line(
        df,
        x=x_axis_df_column,
        y=y_axis_df_column,
        title=plot_title,
        color_discrete_sequence=[line_color],
    )

    fig.update_layout(
        title={
            "text": plot_title.upper(),
            "y": 0.875,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        title_font=dict(
            size=16,
            color="black",
            family="Verdana, sans-serif",
        ),
        xaxis_title=x_axis_df_column.upper(),
        yaxis_title=y_axis_df_column.upper(),
        font=dict(size=12, color="black", family="Verdana, sans-serif"),
    )
    if save_visualization:
        output_directory.mkdir(parents=True, exist_ok=True)
        pio.write_image(
            fig,
            f"{output_directory}/{convert_string_to_snake_case(plot_title)}.{img_format}",
        )
    else:
        fig.show()


def interactive_visualization_from_multiple_dataframes(dfs: Dict[str, pd.DataFrame], plot_title: str, x_axis_df_column: str, y_axis_df_column: str) -> None:
    fig = go.Figure()

    # Add traces for each dataframe
    for df_name, df in dfs.items():
        fig.add_trace(go.Scatter(x=df[x_axis_df_column], y=df[y_axis_df_column], name=df_name))

    fig.show()