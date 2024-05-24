from pathlib import Path

def get_first_element(filename: str, split_character: str='_') -> str:
    """
    Returns the first element of a filename string after splitting it by a specified character.

    Parameters:
    filename (str): The filename string to split.
    split_character (str, optional): The character to split the filename string by. Default is '_'.

    Returns:
    str: The first element of the filename string.

    """
    return filename.split(split_character)[0]


def get_first_element_from_filename(file_path: Path, split_character) -> str:
    """
    Extracts the first element from the filename of a given file path.

    Parameters:
        file_path (Path): The path to the file.
        split_character (str): The character used to split the filename.

    Returns:
        str: The first element of the filename.

    """
    return file_path.name.split(split_character)[0]

def convert_string_to_snake_case(input_string: str) -> str:
    """
    Converts a string to snake case by replacing spaces with underscores and converting to lowercase.

    Parameters:
        input_string (str): The input string to be converted.

    Returns:
        str: The converted string in snake case.
    """
    return input_string.lower().replace(" ", "_")

def get_tqdm_parameters():
    """
    Returns a dictionary of parameters for configuring tqdm progress bar.

    Returns:
        dict: A dictionary containing the following parameters:
            - desc (str): Description of the progress bar.
            - leave (bool): Whether to leave the progress bar on the screen after completion.
            - unit (str): Unit of measurement for the progress bar.
            - unit_scale (bool): Whether to scale the unit of measurement.
    """
    return {
        "desc": "Processing images",
        "leave": True,
        "unit": " image",
        "unit_scale": True,
    }