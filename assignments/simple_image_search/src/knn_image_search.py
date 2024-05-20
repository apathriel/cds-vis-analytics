from pathlib import Path
from typing import Generator, List, Tuple, Dict, Union

import numpy as np
from numpy.linalg import norm

import pandas as pd

from image_selection_gui import ImageSelectionGUI
from utilities import (
    get_full_paths,
    select_random_files,
    image_generator,
    write_dict_to_csv,
)

from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm as tqdm_bar

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from visualize_results import visualize_image_search_similarity


def extract_image_features(
    img_path: Path,
    pretrained_model: Model,
    image_shape: Tuple[int, int, int] = (224, 224, 3),
    return_normalized: bool = True,
) -> np.ndarray:
    """
    Extract features from image data using a pretrained model.

    Parameters:
        img_path (Path): The path to the image file.
        pretrained_model (Model): The pretrained model to use for feature extraction.
        image_shape (Tuple[int, int, int], optional): The desired shape of the input image. Defaults to (224, 224, 3).
        return_normalized (bool, optional): Whether to return normalized features. Defaults to True.

    Returns:
        np.ndarray: The extracted image features.

    """
    # Define input image shape for the model
    input_shape = image_shape
    # Load image from file path
    img = load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    # Convert image to array. Required input format for the pretrained model processing
    img_array = img_to_array(img)
    # Expand to fit dimensions required by the pretrained model
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # Preprocess image
    preprocessed_img = preprocess_input(expanded_img_array)
    # Use the predict function to create feature representation
    features = pretrained_model.predict(preprocessed_img, verbose=False)
    # Flatten the features to a 1D array
    flattened_features = features.flatten()
    # Normalize features
    normalized_features = flattened_features / norm(features)
    return normalized_features if return_normalized else flattened_features


def knn_image_search_pipeline(
    dataset_path: Path, model: Model
) -> Dict[str, np.ndarray]:
    """
    Extract features from all images in a folder using a given model.

    Parameters:
        dataset_path (Path): The path to the folder containing the images.
        model (Model): The model used to extract image features.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping image names to their extracted features.
    """
    image_gen = image_generator(dataset_path)

    dataset_image_features = {}
    for img_path in tqdm_bar(image_gen, desc="Processing images"):
        features = extract_image_features(img_path, model)
        dataset_image_features[img_path.name] = features
    return dataset_image_features


def get_top_5_most_similar_images(
    image_features: Dict[str, Union[np.ndarray, List[float]]],
    neighbors: NearestNeighbors,
    selected_image: Path,
    save_results_to_csv: bool = False,
    output_path: Path = None,
) -> Dict[str, float]:
    # Get the features of the selected image
    selected_image_features = image_features[selected_image.name]
    # Use the features of the selected image in kneighbors
    distances, indices = neighbors.kneighbors([selected_image_features])
    # Get the filenames from the image_features dictionary
    filenames = list(image_features.keys())
    # Create a dictionary where the keys are the filenames and the values are the distances
    top_5_dict = {filenames[indices[0][i]]: distances[0][i] for i in range(1, 6)}

    if save_results_to_csv:
        write_dict_to_csv(top_5_dict, output_path, selected_image.stem)

    return top_5_dict


def gui_get_selected_image(flower_dataset_path: Path) -> Path:
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


import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go

from PIL import Image


def visualize_image_features(
    image_features: Dict[str, np.ndarray], image_directory: Path
) -> None:
    # Convert the image features to a 2D array
    features = np.array(list(image_features.values()))
    # Use PCA to reduce the dimensionality to 2
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    # Create a DataFrame of the 2D features
    df = pd.DataFrame(reduced_features, columns=["x", "y"])
    # Convert the image filenames to PIL Image objects
    df["image"] = [
        Image.open(image_directory / filename) for filename in image_features.keys()
    ]

    # Create a scatter plot of the 2D features
    fig = go.Figure()

    # Add scatter trace for image oints
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(size=8, color="rgba(0, 0, 0, 0)"),  # Make markers invisible
            hoverinfo="none",
        )
    )

    # Add image trace for images
    for i in tqdm_bar(range(len(df)), desc='Adding images to plot'):
        fig.add_layout_image(
            dict(
                source=df["image"][i],  # PIL Image object
                x=df["x"][i],
                y=df["y"][i],
                sizex=1,  # size in plot units
                sizey=1,  # size in plot units
                xanchor="center",
                yanchor="middle",
            )
        )

    fig.update_layout_images(dict(xref="x", yref="y", layer="below"))

    fig.show()


def main():
    # Instantiate the VGG16 model
    model = VGG16(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
    )

    # Initialize Paths for input/output directories
    flower_dataset_path = Path(__file__).parent / ".." / "in"
    csv_output_path = Path(__file__).parent / ".." / "out" / "knn" / "csv"
    plot_output_path = Path(__file__).parent / ".." / "out" / "knn" / "plots"

    # Get the selected image from the GUI
    selected_image = gui_get_selected_image(flower_dataset_path)

    # Extract features from all images in the dataset
    image_features = knn_image_search_pipeline(flower_dataset_path, model)

    # Fit the NearestNeighbors model on the extracted image features
    neighbors = NearestNeighbors(
        n_neighbors=10, algorithm="brute", metric="cosine"
    ).fit(list(image_features.values()))

    # Create a dictionary where the keys are the filenames and the values are the distances
    top_5_most_similar_images = get_top_5_most_similar_images(
        image_features,
        neighbors,
        selected_image,
        save_results_to_csv=True,
        output_path=csv_output_path,
    )

    # Visualize the target image and the 5 most similar images
    visualize_image_search_similarity(
        image_directory=flower_dataset_path,
        target_image=selected_image,
        similar_images=top_5_most_similar_images,
        save_visualization=True,
        output_path=plot_output_path,
    )

    visualize_image_features(image_features=image_features, image_directory=flower_dataset_path)


if __name__ == "__main__":
    main()
