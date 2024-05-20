from pathlib import Path
from tkinter import *
from typing import Dict

from tqdm import tqdm as tqdm_bar

from image_selection_gui import ImageSelectionGUI
from utilities import *
from visualize_results import visualize_image_search_similarity

TQDM_PARAMS = get_tqdm_parameters()


@timing_decorator
def compare_images_in_dataset(
    dataset_path: Path,
    target_image: Path,
    output_path: Path,
    save_top_5_result_to_csv: bool = False,
    convert_to_greyscale: bool = False,
) -> Dict[str, float]:
    """Compare a target image with images in a dataset.

    Args:
        dataset_path (Path): The path to the dataset directory.
        target_image (Path): The path to the selected target image file.
        output_path (Path): The path to the output directory.
        save_top_5_result_to_csv (bool, optional): Whether to save the top 5 most similar images to a CSV file. Defaults to False.
        convert_to_greyscale (bool, optional): Whether to convert the images to greyscale before comparison. Defaults to False.

    Returns:
        Dict[str, float]: A dictionary containing the filenames of the top 5 most similar images as keys and their histogram comparison values as dict values.
    """

    target_image_output_name = target_image.stem + "_most_similar_images"
    target_image_filename = target_image.name
    target_image = load_cv2_image(str(target_image))

    if convert_to_greyscale:
        target_image = convert_image_to_greyscale(target_image)
    target_image_hist = calculate_histogram(target_image)

    most_similar_images_5 = {}

    image_gen = image_generator(dataset_path)

    for image in tqdm_bar(image_gen, **TQDM_PARAMS):
        if image.name == target_image_filename:
            continue

        image_to_be_compared = load_cv2_image(str(image))
        if convert_to_greyscale:
            image_to_be_compared = convert_image_to_greyscale(image_to_be_compared)
        image_to_be_compared_hist = calculate_histogram(image_to_be_compared)

        comparison_val = compare_histograms(
            target_image_hist, image_to_be_compared_hist
        )

        if len(most_similar_images_5) < 5:
            most_similar_images_5.update({image.name: comparison_val})
        elif comparison_val < get_highest_value_from_dict(most_similar_images_5):
            most_similar_images_5.pop(
                max(most_similar_images_5, key=most_similar_images_5.get)
            )
            most_similar_images_5.update({image.name: comparison_val})

    if save_top_5_result_to_csv:
        output_path.mkdir(parents=True, exist_ok=True)
        write_dict_to_csv(most_similar_images_5, output_path, target_image_output_name)

    return most_similar_images_5


def gui_get_selected_image(flower_dataset_path: Path) -> Path:
    """
    Tkinter GUI function to get the selected image input from the user. Wraps class functionality.

    Parameters:
        flower_dataset_path (Path): The path to the flower dataset.

    Returns:
        Path: The path of the selected image.

    """
    # Pseudo random sampling of 3 images from the dataset for GUI selection
    selected_flowers_file_names = select_random_files(flower_dataset_path)
    # Get full paths of the randomly selected images
    selected_flowers = get_full_paths(flower_dataset_path, selected_flowers_file_names)
    # Instantiate the ImageSelectionGUI class instance
    flower_selection_gui = ImageSelectionGUI(
        selected_flowers, selected_flowers_file_names
    )
    # Create the GUI for flower selection. Prompt user for image to compare against dataset
    flower_selection_gui.create_flower_selection_gui()
    # Get the selected image from the GUI. Script operation will not continue execution until an image is selected.
    return flower_selection_gui.get_selected_image()


def main():
    # Initialize Paths for input/output directories
    flower_dataset_path = Path(__file__).parent / ".." / "in"
    csv_output_path = Path(__file__).parent / ".." / "out" / "histogram" / "csv"
    plot_output_path = Path(__file__).parent / ".." / "out" / "histogram" / "plots"

    # Get the selected image from the GUI
    selected_image = gui_get_selected_image(flower_dataset_path)

    # Compare the selected image against the dataset and return the top 5 most similar images
    top_5_most_similar_images = compare_images_in_dataset(
        dataset_path=flower_dataset_path,
        target_image=selected_image,
        output_path=csv_output_path,
        save_top_5_result_to_csv=True,
    )

    visualize_image_search_similarity(
        image_directory=flower_dataset_path,
        target_image=selected_image,
        similar_images=top_5_most_similar_images,
        save_visualization=True,
        output_path=plot_output_path,
    )


if __name__ == "__main__":
    main()
