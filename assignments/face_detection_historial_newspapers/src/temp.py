from collections import Counter
import os
from pathlib import Path
import timeit

from facenet_pytorch import MTCNN, InceptionResnetV1
from logger_utils import get_logger
from PIL import Image
import torch

logger = get_logger(__name__)

newspaper_input_directory = Path(__file__).parent / '..' / 'in'
example_image_path = Path(newspaper_input_directory / 'GDL' / 'GDL-1798-02-05-a-p0001.jpg')

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        print(f"{func.__name__} took {end - start} seconds to run")
        return result
    return wrapper

@timing_decorator
def summarize_file_types(directory: Path) -> None:
    file_types = Counter()

    for file in Path(directory).rglob('*'):
        if file.is_file():
            file_types[file.suffix] += 1

    for file_type, count in file_types.items():
        print(f'{file_type}: {count}')


def get_num_files_in_directory(directory: Path, file_type: str = None) -> int:
    if file_type:
        return len(list(directory.glob(f'*.{file_type}')))
    else:
        return len(list(directory.glob('*')))

for directory in newspaper_input_directory.glob('*'):
        num_files = get_num_files_in_directory(directory)
        print(f'{directory.name}: {num_files} files')
        for image_path in directory.glob('*.jpg'):
            newspaper_image = Image.open(image_path)
            face_bounding_box_positions, _ = mtcnn.detect(newspaper_image)
            number_of_faces = len(face_bounding_box_positions)

summarize_file_types(newspaper_input_directory)