from collections import Counter
from pathlib import Path
from typing import List, Union, Optional, Dict

import glob
import pandas as pd

from code_utilities import get_logger

logger = get_logger(__name__)


def summarize_file_types(directory: Path) -> None:
    """
    Summarizes the file types in a given directory.

    Parameters:
        directory (Path): The directory to analyze.

    """
    file_types = Counter()

    for file in Path(directory).rglob("*"):
        if file.is_file():
            file_types[file.suffix] += 1

    for file_type, count in file_types.items():
        print(f"{file_type}: {count}")



def get_num_files_in_directory(directory: Path, file_type: str = None) -> int:
    """
    Count the number of files in a directory.

    Parameters:
        directory (Path): The directory to search for files.
        file_type (str, optional): The file extension to filter by. Defaults to None.

    Returns:
        int: The number of files in the directory.

    """
    if file_type:
        return len(list(directory.glob(f"*.{file_type}")))
    else:
        return len(list(directory.glob("*")))


def extract_metadata_from_filename(file_path: Path) -> dict:
    """
    Extracts specific metadata from a given file path.

    Parameters:
        file_path (Path): The path of the file.

    Returns:
        dict: A dictionary containing the extracted metadata, including the file name, newspaper name,
              decade, month, date, and page.

    Raises:
        ValueError: If the filename format is invalid.
        Exception: If an unexpected error occurs while extracting metadata from the filename.
    """
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


def export_df_as_csv(df: pd.DataFrame, directory: Path, filename: str) -> None:
    """
    Export a pandas DataFrame as a CSV file.

    Paramters:
        df (pd.DataFrame): The DataFrame to be exported.
        directory (Path): The directory where the CSV file will be saved.
        filename (str): The name of the CSV file.

    Raises:
        PermissionError: If the function does not have permission to create the directory or write the file.
        OSError: If an OS error occurs when trying to create the directory.
        Exception: If an unexpected error occurs when trying to write to the file.
    """
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


def load_csv_as_df_from_directory(
    directory: Path, return_filenames: bool = False
) -> Union[Dict[str, pd.DataFrame], List[pd.DataFrame]]:
    """
    Load CSV files from a directory into pandas DataFrames.

    Parameters:
        directory (Path): The directory path where the CSV files are located.
        return_filenames (bool, optional): Whether to return a dictionary with filenames as keys and DataFrames as values.
            If False, a list of DataFrames will be returned. Defaults to False.

    Returns:
        Union[Dict[str, pd.DataFrame], List[pd.DataFrame]]: A dictionary or list of pandas DataFrames containing the loaded CSV data.

    Raises:
        FileNotFoundError: If the specified directory is not found.
        Exception: If an unexpected error occurs while loading the CSV files.

    """
    try:
        # Get a list of all CSV files in the directory
        csv_files = glob.glob(f"{directory}/*.csv")

        # Load each CSV file into a DataFrame and store in a dictionary or list
        if return_filenames:
            dataframes = {Path(file).name: pd.read_csv(file) for file in csv_files}
        else:
            dataframes = [pd.read_csv(file) for file in csv_files]

        return dataframes
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading CSV files from directory: {e}")
        return []


def get_df_by_newspaper_facial_recognition_metrics(
    df: pd.DataFrame, face_percentage_column_title: str
) -> pd.DataFrame:
    """
    Calculate facial recognition metrics for each decade based on the given DataFrame.

    Paramters:
        df (pd.DataFrame): The input DataFrame containing the data.
        face_percentage_column_title (str): The title of the column to store the calculated face percentage.

    Returns:
        pd.DataFrame: The resulting DataFrame with the calculated metrics.

    """
    # Calculate total number of faces and total number of pages for each decade
    total_faces_and_pages = (
        df.groupby("Decade").agg({"Num Faces": "sum", "Page": "count"}).reset_index()
    )

    # Calculate number of pages that contain at least one face for each decade
    num_pages_with_faces = (
        df[df["Num Faces"] > 0].groupby("Decade").agg({"Page": "count"}).reset_index()
    )

    # Create a DataFrame that includes all possible decades
    all_decades = pd.DataFrame(
        {"Decade": range(df["Decade"].min(), df["Decade"].max() + 1)}
    )

    # Merge the three dataframes on "Decade"
    df = all_decades.merge(total_faces_and_pages, on="Decade", how="left").merge(
        num_pages_with_faces,
        on="Decade",
        how="left",
        suffixes=("_total", "_with_faces"),
    )
    # Fill NaN values with 0
    df.fillna(0, inplace=True)

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
