from pathlib import Path
from typing import Dict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def visualize_image_search_similarity(
    image_directory: Path,
    target_image: Path,
    similar_images: Dict[str, float],
    save_visualization: bool = False,
    output_path: Path = None,
) -> None:
    """Visualize the target image and the 5 most similar images.

    Args:
        image_directory (Path): The path to the directory containing the images.
        target_image (Path): The path to the target image.
        similar_images (Dict[str, float]): A dictionary containing the paths of the 5 most similar images as keys and their similarity scores as values.
        save_visualization (bool, optional): Whether to save the visualization as an image. Defaults to False.
        output_path (Path, optional): The path to save the visualization image. Defaults to None.
    """
    # Create a GridSpec with 2 rows and 5 columns for easier image placement
    gs = gridspec.GridSpec(2, 5, height_ratios=[1, 1])

    # Create a subplot for the target image in the first row, spanning all columns
    ax = plt.subplot(gs[0, :])
    img = mpimg.imread(str(target_image))
    ax.imshow(img)
    ax.set_title(f"Target Image - {target_image.stem}")
    ax.axis("off")

    # Sort the similar_images dictionary by values (scores) in descending order
    similar_images = dict(sorted(similar_images.items(), key=lambda item: item[1], reverse=True))

    # Create subplots for the similar images in the second row
    for i, (image_path, score) in enumerate(similar_images.items()):
        ax = plt.subplot(gs[1, i])
        
        img = mpimg.imread(str(image_directory / image_path))
        ax.imshow(img)
        ax.set_title(f"Score: {score:.2f}", fontsize=8)  # Set the title to the similarity score
        ax.set_xlabel(image_path, fontsize=8)  # Set the x-label to the image file name
        ax.xaxis.set_label_position("top")  # Set the x-label position to top
        ax.xaxis.tick_top()  # Set the x-ticks position to top
        ax.tick_params(axis="x", which="both", length=0)  # Hide x-ticks
        ax.tick_params(axis="y", which="both", length=0)  # Hide y-ticks
        ax.set_xticklabels([])  # Hide x-tick labels
        ax.set_yticklabels([])  # Hide y-tick labels
        for spine in ["left", "right", "top", "bottom"]:
            ax.spines[spine].set_visible(False)  # Hide the spine

    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    if save_visualization:
        if output_path is None:
            output_path = Path.cwd()
        output_path.mkdir(parents=True, exist_ok=True)
        # Save the figure with a higher resolution
        plt.savefig(output_path / f"{target_image.stem}_similarity.png", dpi=300)
        print(f"[SYSTEM] Image similarity visualization saved to {output_path}")
    else:
        plt.show()
