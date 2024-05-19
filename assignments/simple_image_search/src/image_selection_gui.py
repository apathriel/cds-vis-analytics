from pathlib import Path
from tkinter import *
import tkinter.ttk as ttk
from typing import List

from PIL import Image, ImageTk


class ImageSelectionGUI:
    """
    A Tkinter GUI class for selecting an image from a list of images.

    Parameters:
        images (List[Path]): A list of image file paths.
        image_titles (List[str]): A list of titles corresponding to the images.

    Attributes:
        images (List[Path]): A list of image file paths.
        image_titles (List[str]): A list of titles corresponding to the images.
        root (Tk): The root Tkinter window.
        mainframe (ttk.Frame): The main frame of the GUI.
        photo_images (List[ImageTk.PhotoImage]): A list to store references to PhotoImage objects.
        selected_image (List[Path]): A shared variable to store the selected image.

    Methods:
        create_image_label: Creates a label with the image and title.
        create_image_button: Creates a button to select the image.
        return_selected_image: Prints the selected image and stores it in the selected_image attribute.
        create_flower_selection_gui: Creates the GUI for image selection.

    Returns:
        Path or None: The selected image file path, or None if no image is selected.
    """

    def __init__(self, images: List[Path], image_titles: List[Path]):
        self.images = images
        self.image_titles = image_titles
        self.root = Tk()
        self.root.title("Simple Image Search")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.photo_images = []  # List to store references to PhotoImage objects
        self.selected_image = []  # Shared variable to store the selected image

    def create_image_label(self, image: Path, title: str, i: int) -> ImageTk.PhotoImage:
        """
        Creates a label with the image and title.

        Parameters:
            image (Path): The image file path.
            title (str): The title of the image.
            i (int): The index of the image in the list.

        Returns:
            ImageTk.PhotoImage: The PhotoImage object representing the resized image.
        """
        title_label = Label(self.mainframe, text=title)
        title_label.grid(row=1, column=i)
        img = Image.open(image)
        resized_img = img.resize((400, 400))
        photo_image = ImageTk.PhotoImage(resized_img)
        image_label = Label(self.mainframe, image=photo_image)
        image_label.grid(row=2, column=i, padx=10)
        return photo_image  # Return the PhotoImage object to keep a reference to it

    def create_image_button(self, image: Path, i: int) -> None:
        """
        Creates a button to select the image.

        Parameters:
            image (Path): The image file path.
            i (int): The index of the image in the list.
        """
        image_button = Button(
            self.mainframe,
            width=16,
            relief="raised",
            text=f"Select image {i+1}",
            pady=5,
            command=lambda: self.select_and_store_image(image),
        )
        image_button.grid(row=3, column=i)

    def select_and_store_image(self, image: Path) -> None:
        """
        Prints the selected image and stores it in the selected_image attribute.

        Parameters:
            image (Path): The selected image file path.
        """
        print(f"[SYSTEM] {image.stem} has been selected!")
        self.selected_image.append(image)
        self.root.destroy()

    def get_selected_image(self) -> Path:
        """
        Returns the selected image file path.

        Returns:
            Path or None: The selected image file path, or None if no image is selected.
        """
        return self.selected_image[0] if self.selected_image else None

    def create_flower_selection_gui(self):
        """
        Creates the GUI for image selection. Calls other necessary instance methods for creating the GUI

        """
        for i, image in enumerate(self.images):
            photo_image = self.create_image_label(image, self.image_titles[i].name, i)
            self.photo_images.append(photo_image)  # Keep a reference to the PhotoImage object
            self.create_image_button(image, i)

        self.root.mainloop()