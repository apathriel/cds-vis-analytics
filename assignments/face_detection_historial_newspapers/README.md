# 📰 Detecting faces in historical newspapers
This project aims to detect faces in historical newspapers through adoption machine learning methodologies. The main goal is to analyze the occurance and frequency of human faces, grouped by newspaper page, in historical newspaper archives from three historic swiss newspapers: Journal de Genève (JDG, 1826-1994); the Gazette de Lausanne (GDL, 1804-1991); and the Impartial (IMP, 1881-2017). The intent behind the project is providing insights into societal trends through computational analysis of cultural data, aligning with digital humanities methodology.

## 📈 Data

### 📋 Dataset
This project utilizes the historical newspaper dataset stemming from paper 'Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers' by Barman et al., (2021). The dataset's individual sub folders, each containing the images for seperate newspapers, should be placed in the `in` directory. The dataset can be sourced from [here](https://zenodo.org/records/3706863). You can download the dataset directly [here](https://zenodo.org/records/3706863/files/images.zip?download=1). The script only looks at .jpg files, so you can leave any other files.

### 🤖 Model
The face detection model used in this project is `facenet-pytorch`, a PyTorch implementation of the deep learning CNN FaceNet face recognition model. This model is capable of detecting faces in images.

The MTCNN (Multi-task Cascaded Convolutional Networks) is initialized for performing face detection. Additionally, embeddings could be extracted using the InceptionResnetV1 model. Embeddings are not utilized in this project. 

## 📂 Project structure
```
└── face_detection_historical_newspapers
    ├── in/
    │    ├── GDL/
    │    ├── IMP/
    │    └── JDG/
    │
    ├── out/
    │    ├── csv_results/
    │    └── plots/
    │
    ├── src/
    │    ├── base_utilities.py
    │    ├── code_utilities.py
    │    ├── data_processing_utilities.py
    │    ├── face_detection.py 
    │    └── plotting_utilities.py
    │        
    ├── README.md
    ├── requirements.txt
    ├── setup_unix.sh
    └── setup_win.sh
```

## ⚙️ Setup
To set up the project, you need to create a virtual environment and install the required packages. You can do this by running the appropriate setup script for your operating system.

### 🐍 Dependencies
Please ensure you have the following dependencies installed on your system:
- **Python**: `version 3.12.3`

### 💾 Installation
1. Clone the repository
```sh
git clone https://github.com/apathriel/cds-vis-analytics
```
2. Navigate to the project directory
```sh
cd assignments
cd face_detection_historical_newspapers
```
3. Run the setup script to install dependencies, depending on OS.
```sh
bash setup_unix.sh
```
4. Activate virtual environment (OS-specific) and run main py scripts.
```sh
source env/bin/activate
python src/face_detection.py
```

## 🚀 Usage
After setting up the project, you can run the main script (src/face_detection.py) to start the face detection process. The results will be stored in the out/ directory.

## 📊 Results

## 📖 References
- [Medium article on facenet-pytorch](https://medium.com/@danushidk507/facenet-pytorch-pretrained-pytorch-face-detection-mtcnn-and-facial-recognition-b20af8771144)
- [Barman et al. - Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers](https://zenodo.org/records/4065271)
- [Datasets and Models for Historical Newspaper Article Segmentation](https://zenodo.org/records/3706863)