from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Dict

from PIL import Image
from PIL import UnidentifiedImageError
import pandas as pd
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm as tqdm_bar

from base_utilities import get_tqdm_parameters

from code_utilities import get_logger, timing_decorator

from data_processing_utilities import (
    extract_metadata_from_filename,
    export_df_as_csv,
    get_df_by_newspaper_facial_recognition_metrics,
    get_num_files_in_directory,
)

from plotting_utilities import (
    construct_visualization_parameters,
    preload_and_visualize_results,
    visualize_trend_by_time_from_df,
)

logger = get_logger(__name__)

# Initialize MTCNN for face detection
FACE_RECOGNITION_MTCNN = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
RESNET = InceptionResnetV1(pretrained="casia-webface").eval()

def newspaper_image_face_recognition_and_extract_metadata(image_path: Path) -> dict:
    """
    Perform face recognition on a newspaper image and extract metadata.

    Parameters:
        image_path (Path): The path to the newspaper image.

    Returns:
        dict: A dictionary containing the extracted metadata, including the number of faces detected.

    Raises:
        OSError: If the image file is corrupted or truncated.
        UnidentifiedImageError: If the image file cannot be identified.

    """
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
    """
    Process images in a directory and create a pandas DataFrame with the extracted metadata.

    Parameters:
        input_directory (Path): The directory containing the images to be processed.
        image_file_format (str): The file format of the images to be processed.
        num_images_to_process (int, optional): The number of images to process. If None, all images in the directory will be processed. Defaults to None.
        face_percentage_column_title (str, optional): The title of the column that will store the face percentage in the resulting DataFrame. Defaults to "".
        tqdm_parameters (Dict, optional): Additional parameters to be passed to the tqdm progress bar. Defaults to {}.
        use_multiprocessing (bool, optional): Whether to use multiprocessing for faster processing. Defaults to False.
        num_workers (int, optional): The number of worker processes to use if multiprocessing is enabled. Defaults to 4.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted metadata from the processed images.
    """
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
    """
    Perform facial recognition on images in the input directory and create a dataframe with the results.

    Parameters:
        input_directory_path (Path): The path to the input directory containing the images.
        results_output_directory_path (Path): The path to the directory where the results will be saved.
        csv_output_directory (str, optional): The name of the directory where the CSV results will be saved. Defaults to "csv_results".
        image_file_format (str, optional): The file format of the images. Defaults to ".jpg".
        visualize_results (bool, optional): Whether to visualize the results. Defaults to True.
        visualization_parameters (Dict, optional): Additional parameters for visualization. Defaults to {}.
        tqdm_parameters (Dict, optional): Additional parameters for the tqdm progress bar. Defaults to {}.
        face_percentage_column_title (str, optional): The title of the column in the CSV file that will contain the face percentage. Defaults to "".
        num_images_to_process (int, optional): The number of images to process. Defaults to None.

    """
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

    TQDM_PARAMS = get_tqdm_parameters()

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

    # Interactive plot of all face detection results by newspaper
    # preload_and_visualize_results(visualization_method="group")

    # Visualize all face detection results by newspaper
    # preload_and_visualize_results()
