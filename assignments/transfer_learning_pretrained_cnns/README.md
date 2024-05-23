# 🕵 Document classification using pretrained image embeddings

## 📈 Data

### 📋 Dataset

### 🤖 Model

## 📂 Project structure
```
└── transfer_learning_pretrained_cnns
    ├── in/
    │
    ├── out/
    │    ├── models/
    │    ├── VGG16_tobacco_plot.pdf
    │    └── VGG16_tobacco_report.txt
    │
    ├── src/
    │    ├── document_classification.py
    │    └── utilities/
    │         ├── data_processing_utilities.py
    │         ├── logging_utilities.py
    │         ├── model_compilation_utilities.py
    │         └── plotting_utilities.py
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
cd transfer_learning_pretrained_cnns
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

### 💻 CLI Reference

### 🧰 Utilities
- ``data_processing_utilities.py``: Contains functions for loading and preprocessing the training data, splitting the data, and binarizing the labels.
- ``model_compilation_utilities.py``: Contains functions for defining the model architecture, compiling the model, and optionally augmenting the training data.
- ``plotting_utilities.py``: Contains functions for plotting the model's training history.
- ``logging_utilities.py``: Contains function for instantiating

## 📊 Results

## 📖 References