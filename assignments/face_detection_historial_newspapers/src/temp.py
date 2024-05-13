from collections import Counter
from itertools import islice
import glob
from pathlib import Path
import timeit
from typing import Dict

from facenet_pytorch import MTCNN, InceptionResnetV1
from logger_utils import get_logger
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from PIL import UnidentifiedImageError
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from tqdm import tqdm as tqdm_bar

logger = get_logger(__name__)

# Initialize MTCNN for face detection
FACE_RECOGNITION_MTCNN = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
RESNET = InceptionResnetV1(pretrained="casia-webface").eval()


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        logger.info(f"{func.__name__} took {end - start} seconds to run")
        return result

    return wrapper


def load_csv_as_df_from_directory(directory: Path) -> pd.DataFrame:
    try:
        # Get a list of all CSV files in the directory
        csv_files = glob.glob(f"{directory}/*.csv")

        # Load each CSV file into a DataFrame and store in a list
        dataframes = [pd.read_csv(file) for file in csv_files]

        return dataframes
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading CSV files from directory: {e}")
        return []


def extract_metadata_from_filename(file_path: Path) -> dict:
    try:
        filename = file_path.name
        newspaper_name, year, month, date, _, page = filename.split("-")
        return {
            "File": filename,
            "Newspaper": newspaper_name,
            "Decade": (int(year) // 10) * 10,
            "Month": month,
            "Date": date,
            "Page": page,
        }
    except ValueError:
        logger.error(f"Invalid filename format: {file_path.name}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error extracting metadata from filename: {e}")
        return {}


def summarize_file_types(directory: Path) -> None:
    file_types = Counter()

    for file in Path(directory).rglob("*"):
        if file.is_file():
            file_types[file.suffix] += 1

    for file_type, count in file_types.items():
        print(f"{file_type}: {count}")


def get_num_files_in_directory(directory: Path, file_type: str = None) -> int:
    if file_type:
        return len(list(directory.glob(f"*.{file_type}")))
    else:
        return len(list(directory.glob("*")))


def convert_string_to_snake_case(input_string: str) -> str:
    return input_string.lower().replace(" ", "_")


def visualize_trend_by_time_from_df(
    df: pd.DataFrame,
    plot_title: str,
    x_axis_df_column: str,
    y_axis_df_column: str,
    save_visualization: bool,
    add_regression: bool = True,
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


def get_df_by_newspaper_facial_recognition_metrics(
    df: pd.DataFrame, face_percentage_column_title: str
) -> pd.DataFrame:
    # Calculate total number of faces and total number of pages for each decade
    total_faces_and_pages = (
        df.groupby("Decade").agg({"Num Faces": "sum", "Page": "count"}).reset_index()
    )

    # Calculate number of pages that contain at least one face for each decade
    num_pages_with_faces = (
        df[df["Num Faces"] > 0].groupby("Decade").agg({"Page": "count"}).reset_index()
    )

    # Merge the two dataframes on "Decade"
    df = pd.merge(
        total_faces_and_pages,
        num_pages_with_faces,
        on="Decade",
        suffixes=("_total", "_with_faces"),
    )

    # Calculate the percentage of pages containing faces
    df[face_percentage_column_title] = df["Page_with_faces"] / df["Page_total"] * 100

    # Calculate the total number of pages across all decades
    total_pages_all_decades = df["Page_total"].sum()

    # Add a new column for the weighted analysis
    df["Weighted Analysis"] = (
        df[face_percentage_column_title] * df["Page_total"] / total_pages_all_decades
    )

    # Rename the columns
    df = df.rename(
        columns={"Page_total": "Total Pages", "Page_with_faces": "Num Pages with Faces"}
    )

    return df


def export_df_as_csv(df: pd.DataFrame, directory: Path, filename: str):
    try:
        # Create the directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied when trying to create directory: {directory}")
        return
    except OSError as e:
        logger.error(f"OS error occurred when trying to create directory: {e}")
        return

    # Create the full file path
    file_path = directory / filename

    try:
        df.to_csv(file_path, index=False)
    except PermissionError:
        logger.error(f"Permission denied when trying to write to file: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred when trying to write to file: {e}")


def newspaper_image_face_recognition_and_extract_metadata(image_path: Path) -> dict:
    try:
        newspaper_image = Image.open(image_path)
        face_bounding_box_positions, _ = FACE_RECOGNITION_MTCNN.detect(newspaper_image)
        number_of_faces = (
            len(face_bounding_box_positions)
            if face_bounding_box_positions is not None
            else 0
        )
        metadata = extract_metadata_from_filename(image_path)
        return {**metadata, "Num Faces": number_of_faces}
    except (OSError, UnidentifiedImageError):
        logger.error(f"Skipping corrupted or truncated image: {image_path.name}")
        return None


def process_images_in_directory_and_create_dataframe(
    directory: Path,
    image_file_format: str,
    num_images_to_process=None,
    face_percentage_column_title: str = "",
) -> pd.DataFrame:
    logger.info(f"Processing directory {directory.name}")

    newspaper_data = []
    num_files = get_num_files_in_directory(directory)
    logger.info(f"{directory.name}: {num_files} files")

    for image_path in islice(
        tqdm_bar(directory.glob(f"*{image_file_format}")), num_images_to_process
    ):
        image_data = newspaper_image_face_recognition_and_extract_metadata(image_path)
        if image_data is not None:
            newspaper_data.append(image_data)

    logger.info(f"Processed {num_files} images from the newspaper {directory.name}")
    df = pd.DataFrame(newspaper_data)
    df = get_df_by_newspaper_facial_recognition_metrics(
        df=df, face_percentage_column_title=face_percentage_column_title
    )
    return df


def construct_visualization_parameters(
    df, color_iteration_index, plot_title, visualization_parameters: Dict
) -> Dict:
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

    visualization_parameters.update(
        {
            "df": df,
            "plot_title": f"{plot_title} - Evolution of faces in newspapers by decade",
            "line_color": colors[color_iteration_index % len(colors)],
        }
    )

    return visualization_parameters


def facial_recognition_by_image_from_directory(
    input_directory_path: Path,
    results_output_directory_path: Path,
    csv_output_directory: str = "csv_results",
    image_file_format: str = ".jpg",
    visualize_results: bool = True,
    visualization_parameters: Dict = {},
    face_percentage_column_title: str = "",
    num_images_to_process: int = None,
):
    logger.info(
        f"Processing sub directories from directory {input_directory_path.name}"
    )

    directories = list(input_directory_path.glob("*"))
    if not directories:
        logger.warning("No directories found in the input directory.")
        return

    for i, directory in enumerate(directories):
        if not directory.is_dir():
            continue
        df = process_images_in_directory_and_create_dataframe(
            directory,
            image_file_format,
            num_images_to_process,
            face_percentage_column_title,
        )
        export_df_as_csv(
            df,
            directory=results_output_directory_path / csv_output_directory,
            filename=f"{directory.name}_face_detection_results.csv",
        )
        if visualize_results:
            updated_visualization_parameters = construct_visualization_parameters(
                df, i, directory.name, visualization_parameters
            )

            visualize_trend_by_time_from_df(**updated_visualization_parameters)


def main():
    NEWSPAPER_INPUT_DIRECTORY = Path(__file__).parent / ".." / "in"
    MAIN_RESULTS_OUTPUT_DIRECTORY = Path(__file__).parent / ".." / "out"

    DECADE_COLUMN_TITLE = "Decade"
    PERCENTAGE_OF_FACES_COLUMN_TITLE = "Percentage of Pages with Faces"

    VISUALIZATION_PARAMS = {
        "x_axis_df_column": DECADE_COLUMN_TITLE,
        "y_axis_df_column": PERCENTAGE_OF_FACES_COLUMN_TITLE,
        "save_visualization": True,
        "output_directory": MAIN_RESULTS_OUTPUT_DIRECTORY / "plots",
        "img_format": "pdf",
    }

    facial_recognition_by_image_from_directory(
        input_directory_path=NEWSPAPER_INPUT_DIRECTORY,
        results_output_directory_path=MAIN_RESULTS_OUTPUT_DIRECTORY,
        face_percentage_column_title=PERCENTAGE_OF_FACES_COLUMN_TITLE,
        visualize_results=True,
        visualization_parameters=VISUALIZATION_PARAMS,
        num_images_to_process=None,
    )


def preload_and_visualize_results():
    df_list = load_csv_as_df_from_directory(
        Path(__file__).parent / ".." / "out" / "csv_results"
    )
    for df in df_list:
        visualize_trend_by_time_from_df(
            df=df,
            plot_title="Face Detection Results",
            x_axis_df_column="Decade",
            y_axis_df_column="Percentage of Pages with Faces",
            save_visualization=False,
            add_regression=False,
        )


if __name__ == "__main__":
    preload_and_visualize_results()
