from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Dict

from facenet_pytorch import MTCNN, InceptionResnetV1
from code_utilities import (
    get_logger,
    timing_decorator
)

from data_processing_utilities import (
    extract_metadata_from_filename,
    export_df_as_csv,
    get_num_files_in_directory,
    load_csv_as_df_from_directory,
)
import pandas as pd
from plotting_utilities import (
    construct_visualization_parameters,
    visualize_trend_by_time_from_df,
)
from PIL import Image
from PIL import UnidentifiedImageError
from tqdm import tqdm as tqdm_bar

logger = get_logger(__name__)

# Initialize MTCNN for face detection
FACE_RECOGNITION_MTCNN = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
RESNET = InceptionResnetV1(pretrained="casia-webface").eval()


def preload_and_visualize_results(
    csv_dir_path: Path = Path(__file__).parent / ".." / "out" / "csv_results",
) -> None:
    df_list = load_csv_as_df_from_directory(csv_dir_path)
    for df in df_list:
        visualize_trend_by_time_from_df(
            df=df,
            plot_title="Face Detection Results",
            x_axis_df_column="Decade",
            y_axis_df_column="Percentage of Pages with Faces",
            save_visualization=False,
            add_regression=False,
        )


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

    # Calculate the minimum + range of the "Weighted Analysis" column
    min_weighted_analysis = df["Weighted Analysis"].min()
    range_weighted_analysis = df["Weighted Analysis"].max() - min_weighted_analysis

    # Normalize the "Weighted Analysis" column
    df["Normalized Weighted Analysis"] = (
        df["Weighted Analysis"] - min_weighted_analysis
    ) / range_weighted_analysis

    # Rename the columns
    df = df.rename(
        columns={"Page_total": "Total Pages", "Page_with_faces": "Num Pages with Faces"}
    )

    return df


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
    input_directory: Path,
    image_file_format: str,
    num_images_to_process=None,
    face_percentage_column_title: str = "",
    tqdm_parameters: Dict = {},
    use_multiprocessing: bool = False,
    num_workers: int = 4,
) -> pd.DataFrame:
    logger.info(f"Processing directory {input_directory.name}")

    num_files = get_num_files_in_directory(input_directory)
    logger.info(f"{input_directory.name}: {num_files} files")

    def image_data_generator():
        image_paths = list(
            islice(input_directory.glob(f"*{image_file_format}"), num_images_to_process)
        )
        if use_multiprocessing:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for image_data in tqdm_bar(
                    executor.map(
                        newspaper_image_face_recognition_and_extract_metadata,
                        image_paths,
                    ),
                    total=len(image_paths),
                    **tqdm_parameters,
                ):
                    if image_data is not None:
                        yield image_data
        else:
            for image_path in tqdm_bar(image_paths, **tqdm_parameters):
                image_data = newspaper_image_face_recognition_and_extract_metadata(
                    image_path
                )
                if image_data is not None:
                    yield image_data

    newspaper_data = image_data_generator()

    df = pd.DataFrame(newspaper_data)
    num_images_processed = len(df)
    logger.info(
        f"All {num_images_processed} images in {input_directory.name} have been processed."
    )
    df = get_df_by_newspaper_facial_recognition_metrics(
        df=df, face_percentage_column_title=face_percentage_column_title
    )
    return df

@timing_decorator
def facial_recognition_newspaper_image_pipeline(
    input_directory_path: Path,
    results_output_directory_path: Path,
    csv_output_directory: str = "csv_results",
    image_file_format: str = ".jpg",
    visualize_results: bool = True,
    visualization_parameters: Dict = {},
    tqdm_parameters: Dict = {},
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
            tqdm_parameters,
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
    newspaper_input_directory = Path(__file__).parent / ".." / "in"
    main_results_output_directory = Path(__file__).parent / ".." / "out"

    decade_column_title = "Decade"
    percentage_of_faces_column_title = "Percentage of Pages with Faces"

    VISUALIZATION_PARAMS = {
        "x_axis_df_column": decade_column_title,
        "y_axis_df_column": percentage_of_faces_column_title,
        "save_visualization": True,
        "output_directory": main_results_output_directory / "plots",
        "img_format": "pdf",
    }

    TQDM_PARAMS = {
        "desc": "Processing images",
        "leave": True,
        "ncols": 100,
        "unit": "image",
        "unit_scale": True,
    }

    facial_recognition_newspaper_image_pipeline(
        input_directory_path=newspaper_input_directory,
        results_output_directory_path=main_results_output_directory,
        face_percentage_column_title=percentage_of_faces_column_title,
        visualize_results=True,
        visualization_parameters=VISUALIZATION_PARAMS,
        tqdm_parameters=TQDM_PARAMS,
        num_images_to_process=None,
    )


if __name__ == "__main__":
    main()
