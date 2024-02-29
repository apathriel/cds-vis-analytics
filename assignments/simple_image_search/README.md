# Building a simple image search algorithm

The project utilizes the OpenCV module to perform a simple image search. The images are compared based on normalized colour histograms, with the (dis)similarity being measured through the Chi-Square metric. The project takes an input, a target image to be compared against the dataset, the output consisting of a .csv file detailing the 5 most similiar images in the dataset. The project utilizes the Tkinter module to construct a simple GUI containing a pseudo randomized sample of the image dataset, allowing for user selection of the target for comparison between the 3 images.

## Project structure
```
└── simple_image_search
	├── data
	│   ├── input
	│   └── output
	├── src
	│   ├── image_search.py
    	│   ├── image_selection.py
    	│   └── utilities.py
	├── setup.sh
	├── requirements.txt
	├── README.md
```

## Setup
***Dependencies***
Please ensure you have the following dependencies installed on your system:
- **Python**: `version 3.12.2`

### Installation
1. Clone the repository
```sh
git clone https://github.com/apathriel/cds-vis-analytics
```
2. Navigate to the project directory
```sh
cd assignments
cd simple_image_search
```
3. Run the setup script to install dependencies
``` sh
bash setup.sh
```
4. Run the main script
```sh
python src/image_search.py
```

### Usage 
The image dataset should be placed in the 'input' directory. The image search algorithm should be run from the image_search.py script. The target is selected through the functions in image_selection.py. Both scripts utilize functions from the utility.py module. The `compare_images_in_dataset()` function takes a dataset input path, the image selected through the TKinter GUI, and an output path for the resulting .csv file.
