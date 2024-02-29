import os
import timeit
import sys
from tkinter import *
from tqdm import tqdm
from utilities import *
from image_selection import create_flower_selection_gui

sys.path.append(os.path.join("..", ".."))

def compare_images_in_dataset(dataset_path, target_image, output_path, convert_to_greyscale=False):
    """Compare a target image with images in a dataset."""
    start_time = timeit.default_timer()

    target_image_output_name = os.path.basename(target_image).split(".")[0] + "_most_similar_images"
    target_image_filename = os.path.basename(target_image)
    target_image = load_cv2_image(target_image)
    if convert_to_greyscale:
        target_image = convert_image_to_greyscale(target_image)
    target_image_hist = calculate_histogram(target_image)

    most_similar_images_5 = {}

    for image in tqdm(os.listdir(dataset_path)):
        if image == target_image_filename:
            continue

        image_to_be_compared = load_cv2_image(os.path.join(dataset_path, image))
        if convert_to_greyscale:
            image_to_be_compared = convert_image_to_greyscale(image_to_be_compared)
        image_to_be_compared_hist = calculate_histogram(image_to_be_compared)

        comparison_val = compare_histograms(target_image_hist, image_to_be_compared_hist)

        if len(most_similar_images_5) < 5:
            most_similar_images_5.update({image: comparison_val})
        elif comparison_val < get_highest_value_from_dict(most_similar_images_5):
            most_similar_images_5.pop(max(most_similar_images_5, key=most_similar_images_5.get))
            most_similar_images_5.update({image: comparison_val})

    write_dict_to_csv(most_similar_images_5, output_path, target_image_output_name)

    elapsed = timeit.default_timer() - start_time
    print(f"Image search: Elapsed time: {elapsed} seconds")


def main():
    flower_dataset_path = os.path.join("..", "data", "input", "flowers")
    csv_output_path = os.path.join("..", "data", "output")

    selected_flowers_file_names = select_random_files(flower_dataset_path)
    selected_flowers = get_full_paths(flower_dataset_path, selected_flowers_file_names)
    selected_image = create_flower_selection_gui(selected_flowers, selected_flowers_file_names)

    compare_images_in_dataset(flower_dataset_path, selected_image, csv_output_path)

if __name__ == "__main__":
    main()