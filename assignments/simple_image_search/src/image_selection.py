from tkinter import *
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import os

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