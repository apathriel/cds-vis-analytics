from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

from base_utilities import convert_string_to_snake_case

from data_processing_utilities import (
    load_csv_as_df_from_directory,
)


def construct_visualization_parameters(
    df, color_iteration_index, plot_title, visualization_parameters: Dict
) -> Dict:
    """
    Constructs visualization parameters for plotting.

    Parameters:
        df: The dataframe containing the data.
        color_iteration_index: The index used to select the color for the plot.
        plot_title: The title of the plot.
        visualization_parameters: The existing visualization parameters.

    Returns:
        A dictionary containing the updated visualization parameters.
    """
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
    """
    Visualizes the trend of a variable over time from a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        plot_title (str): The title of the plot.
        x_axis_df_column (str): The column name for the x-axis values.
        y_axis_df_column (str): The column name for the y-axis values.
        save_visualization (bool): Whether to save the visualization or display it.
        add_regression (bool, optional): Whether to add a regression line to the plot. Defaults to False.
        output_directory (Path, optional): The directory to save the visualization. Defaults to the parent directory of the current file.
        img_format (str, optional): The format of the saved image. Defaults to "pdf".
        line_color (str, optional): The color of the plot line. Defaults to "blue".

    """
    plt.figure(figsize=(10, 6))

    if add_regression:
        plot_title = f"{plot_title} regression"
        sns.regplot(x=df[x_axis_df_column], y=df[y_axis_df_column], color=line_color)
    else:
        plt.plot(df[x_axis_df_column], df[y_axis_df_column], color=line_color)

    plt.title(plot_title.upper(), fontsize=16, color="black")
    plt.xlabel(x_axis_df_column.upper(), fontsize=12, color="black")
    plt.ylabel(y_axis_df_column.upper(), fontsize=12, color="black")
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
    """
    Generate an interactive trend visualization by time from a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        plot_title (str): The title of the plot.
        x_axis_df_column (str): The column name in the DataFrame to be used as the x-axis.
        y_axis_df_column (str): The column name in the DataFrame to be used as the y-axis.
        save_visualization (bool): Whether to save the visualization as an image file.
        output_directory (Path, optional): The directory to save the visualization image file. Defaults to the parent directory of the current file.
        img_format (str, optional): The format of the saved image file. Defaults to "pdf".
        line_color (str, optional): The color of the line in the plot. Defaults to "blue".

    """
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
            family="sans-serif",
        ),
        xaxis_title=x_axis_df_column.upper(),
        yaxis_title=y_axis_df_column.upper(),
        font=dict(size=12, color="black", family="sans-serif"),
    )
    if save_visualization:
        output_directory.mkdir(parents=True, exist_ok=True)
        pio.write_image(
            fig,
            f"{output_directory}/{convert_string_to_snake_case(plot_title)}.{img_format}",
        )
    else:
        fig.show()


def interactive_visualization_from_multiple_dataframes(
    dfs: Dict[str, pd.DataFrame],
    plot_title: str,
    x_axis_df_column: str,
    y_axis_df_column: str,
) -> None:
    """
    Generate an interactive visualization from multiple dataframes.

    Parameters:
        dfs (Dict[str, pd.DataFrame]): A dictionary of dataframes, where the keys are the names of the dataframes
            and the values are the corresponding pandas DataFrames.
        plot_title (str): The title of the plot.
        x_axis_df_column (str): The column name from the dataframes to be used as the x-axis values.
        y_axis_df_column (str): The column name from the dataframes to be used as the y-axis values.

    """
    fig = go.Figure()

    # Add traces for each dataframe
    for df_name, df in dfs.items():
        fig.add_trace(
            go.Scatter(x=df[x_axis_df_column], y=df[y_axis_df_column], name=df_name)
        )

    fig.show()


def preload_and_visualize_results(
    csv_dir_path: Path = Path(__file__).parent / ".." / "out" / "csv_results",
    visualization_method: str = None,
) -> None:
    """
    Preloads CSV files from a directory and visualizes the results based on the specified visualization method.

    Parameters:
        csv_dir_path (Path, optional): The directory path where the CSV files are located. Defaults to the 'csv_results' directory in the parent directory of the current file.
        visualization_method (str, optional): The visualization method to use. Valid options are 'interactive' and 'group'. Defaults to None.

    Returns:
        None
    """
    df_list = load_csv_as_df_from_directory(csv_dir_path)

    if visualization_method == "interactive":
        for df in df_list:
            interactive_visualization_trend_by_time_from_df(
                df=df,
                plot_title="Face Detection Results",
                x_axis_df_column="Decade",
                y_axis_df_column="Percentage of Pages with Faces",
                save_visualization=False,
            )
    elif visualization_method == "group":
        df_list = load_csv_as_df_from_directory(csv_dir_path, return_filenames=True)
        print(df_list)
        interactive_visualization_from_multiple_dataframes(
            dfs=df_list,
            plot_title="Face Detection Results",
            x_axis_df_column="Decade",
            y_axis_df_column="Percentage of Pages with Faces",
        )
    else:
        for df in df_list:
            visualize_trend_by_time_from_df(
                df=df,
                plot_title="Face Detection Results",
                x_axis_df_column="Decade",
                y_axis_df_column="Percentage of Pages with Faces",
                save_visualization=False,
                add_regression=False,
            )
