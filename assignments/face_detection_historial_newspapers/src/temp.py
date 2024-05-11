from collections import Counter
from itertools import islice
from pathlib import Path
import timeit

from facenet_pytorch import MTCNN, InceptionResnetV1
from logger_utils import get_logger
import pandas as pd
from PIL import Image
from PIL import UnidentifiedImageError
import plotly.express as px
from tqdm import tqdm as tqdm_bar

logger = get_logger(__name__)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        logger.info(f"{func.__name__} took {end - start} seconds to run")
        return result

    return wrapper


def extract_metadata_from_filename(file_path: Path) -> dict:
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


@timing_decorator
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


def df_by_newspaper_facial_recognition_metrics(df: pd.DataFrame) -> pd.DataFrame:
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
    df["Percentage of Pages with Faces"] = (
        df["Page_with_faces"] / df["Page_total"] * 100
    )

    # Rename the columns
    df = df.rename(
        columns={"Page_total": "Total Pages", "Page_with_faces": "Num Pages with Faces"}
    )

    return df


def export_df_as_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, index=False)


@timing_decorator
def process_images_from_directory(
    input_directory_path: Path,
    image_file_format: str = ".jpg",
    num_images_to_process=None,
):
    logger.info(
        f"Processing sub directories from directory {input_directory_path.name}"
    )
    for directory in input_directory_path.glob("*"):
        if not directory.is_dir():
            continue
        logger.info(f"Processing directory {directory.name}")

        newspaper_data = []
        num_files = get_num_files_in_directory(directory)
        logger.info(f"{directory.name}: {num_files} files")

        for image_path in islice(
            tqdm_bar(directory.glob(f"*{image_file_format}")), num_images_to_process
        ):
            try:
                newspaper_image = Image.open(image_path)
                face_bounding_box_positions, _ = mtcnn.detect(newspaper_image)
                number_of_faces = (
                    len(face_bounding_box_positions)
                    if face_bounding_box_positions is not None
                    else 0
                )
                metadata = extract_metadata_from_filename(image_path)
                newspaper_data.append({**metadata, "Num Faces": number_of_faces})
            except (OSError, UnidentifiedImageError):
                logger.error(
                    f"Skipping corrupted or truncated image: {image_path.name}"
                )
                continue
        logger.info(
            f"Processed {num_files} images from the newspaper {directory.name}"
        )
        df = pd.DataFrame(newspaper_data)
        df = df_by_newspaper_facial_recognition_metrics(df)
        export_df_as_csv(
            df,
            results_output_directory / f"{directory.name}_face_detection_results.csv",
        )


# Initialize input directory Path object
newspaper_input_directory = Path(__file__).parent / ".." / "in"
results_output_directory = Path(__file__).parent / ".." / "out"

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained="casia-webface").eval()

process_images_from_directory(newspaper_input_directory)
