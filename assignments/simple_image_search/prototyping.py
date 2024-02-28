# need add utility module
# maybe end GUI

import os
import sys
import random
import pandas as pd
import cv2
from tkinter import *
import tkinter.ttk as ttk
from PIL import Image, ImageTk

sys.path.append(os.path.join("..", ".."))

def get_full_paths(directory, files):
    """Return full paths of files in a directory."""
    return [os.path.join(directory, file) for file in files]


def select_random_files(directory, num_to_select=3):
    """Select random files from a directory."""
    files_in_dir = os.listdir(directory)
    return random.sample(files_in_dir, num_to_select)


def create_image_label(mainframe, image, title, i):
    """Create a label with an image and a title."""
    title_label = Label(mainframe, text=title)
    title_label.grid(row=1, column=i)
    img = Image.open(image)
    resized_img = img.resize((400, 400))
    photo_image = ImageTk.PhotoImage(resized_img)
    image_label = Label(mainframe, image=photo_image)
    image_label.grid(row=2, column=i, padx=10)
    return photo_image  # Return the PhotoImage object to keep a reference to it


def create_image_button(mainframe, image, root, selected_image, i):
    """Create a button for image selection."""
    image_button = Button(mainframe, width=16, relief="raised", text=f"Select image {i+1}", pady=5, 
                          command=lambda: return_selected_image(image, root, selected_image))
    image_button.grid(row=3, column=i)


def return_selected_image(image, root, selected_image):
    """Handle image selection."""
    print(f"[SYSTEM] {os.path.basename(image).split('.')[0]} has been selected!")
    selected_image.append(image)
    root.destroy()


def create_flower_selection_gui(images, image_titles):
    """Create a GUI for flower selection."""
    root = Tk()
    root.title("Simple Image Search")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

    photo_images = []  # List to store references to PhotoImage objects
    selected_image = []  # Shared variable to store the selected image

    for i, image in enumerate(images):
        photo_image = create_image_label(mainframe, image, image_titles[i], i)
        photo_images.append(photo_image)  # Keep a reference to the PhotoImage object
        create_image_button(mainframe, image, root, selected_image, i)

    root.mainloop()

    return selected_image[0] if selected_image else None


def convert_image_to_greyscale(image):
    """Convert an image to greyscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def calculate_histogram(image, convert_to_greyscale=False):
    """Calculate the histogram of an image."""
    if convert_to_greyscale:
        grey_scale_image = convert_image_to_greyscale(image)
        hist = cv2.calcHist([grey_scale_image], [0], None, [256], [0, 256])
    else:
        hist = cv2.calcHist([image], [0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)


def compare_histograms(hist_01, hist_02):
    """Compare two histograms."""
    return round(cv2.compareHist(hist_01, hist_02, cv2.HISTCMP_CHISQR), 2)


def load_cv2_image(path):
    """Load an image using cv2."""
    return cv2.imread(path)


def get_highest_value_from_dict(dictionary):
    """Get the highest value from a dictionary."""
    return max(dictionary.values())


def write_dict_to_csv(dictionary, output_path, filename="output.csv"):
    """Write a dictionary to a CSV file."""
    df = pd.DataFrame(list(dictionary.items()), columns=["Filename", "Distance"])
    df.to_csv(os.path.join(output_path, filename) + ".csv", index=False)
    print(f"[SYSTEM] CSV file containing the 5 most similar images has been written to {output_path}")


def compare_images_in_dataset(dataset_path, target_image, output_path, convert_to_greyscale=False):
    """Compare a target image with images in a dataset."""
    target_image_output_name = os.path.basename(target_image).split(".")[0] + "_most_similar_images"
    target_image_filename = os.path.basename(target_image)
    target_image_hist = calculate_histogram(load_cv2_image(target_image))

    most_similar_images_5 = {}

    for image in os.listdir(dataset_path):
        if image == target_image_filename:
            continue

        image_to_be_compared = calculate_histogram(load_cv2_image(os.path.join(dataset_path, image)))
        comparison_val = compare_histograms(target_image_hist, image_to_be_compared)

        if len(most_similar_images_5) < 5:
            most_similar_images_5.update({image: comparison_val})
        elif comparison_val < get_highest_value_from_dict(most_similar_images_5):
            most_similar_images_5.pop(max(most_similar_images_5, key=most_similar_images_5.get))
            most_similar_images_5.update({image: comparison_val})

    write_dict_to_csv(most_similar_images_5, output_path, target_image_output_name)


def main():
    """Main function."""
    flower_dataset_path = os.path.join("data", "input", "flowers")
    csv_output_path = os.path.join("data", "output")

    selected_flowers_file_names = select_random_files(flower_dataset_path)
    selected_flowers = get_full_paths(flower_dataset_path, selected_flowers_file_names)
    selected_image = create_flower_selection_gui(selected_flowers, selected_flowers_file_names)

    compare_images_in_dataset(flower_dataset_path, selected_image, csv_output_path)


if __name__ == "__main__":
    main()