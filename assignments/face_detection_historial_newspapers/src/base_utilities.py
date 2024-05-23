from pathlib import Path

def get_first_element(filename: str, split_character: str='_') -> str:
    return filename.split(split_character)[0]

def get_first_element_from_filename(file_path: Path, split_character) -> str:
    return file_path.name.split(split_character)[0]

def convert_string_to_snake_case(input_string: str) -> str:
    return input_string.lower().replace(" ", "_")

def get_tqdm_parameters():
    return {
        "desc": "Processing images",
        "leave": True,
        "unit": " image",
        "unit_scale": True,
    }