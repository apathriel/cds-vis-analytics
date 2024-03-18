# Classification benchmarks with Logistic Regression and Neural Networks

The project performs image classification using the cifar10 dataset, downloaded and imported through the TensorFlow module. Image data is preprocessed. Specifically, images are greyscaled, pixel values are normalized, and np.ndarray is reshaped. Numerical labels are converted to lexical, taken from cifar10 docs. Classifier is trained and fit to the data. Predictions on test split are performed and classification is output to `out` directory. The project is comprised of two scripts, both utilizing scikit-learn for machine learning. The script `logistic_regression.py` utilizes the `LogisticRegression()` classifier, while the `neural_network.py` script utilizes the `MLPClassifier()`. In the case of the neural network, the loss curve is plotted and saved.

## Project structure
```
└── image_classification_benchmarks
	├── out
	│   ├── logistic_regression
	│   └── neural_network
	├── src
	│   ├── logistic_regression.py
   	│   └── neural_network.py
	│
	├── setup.sh
	├── run_log.sh
	├── run_neural.sh
	├── requirements.txt
	└── README.md
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
cd image_classification_benchmarks
```
3. Run the setup script to instantiate virtual environment and install dependencies
``` sh
bash setup.sh
```
4. Activate virtual environment, or run main py scripts through run_log and run_neural
```sh
python src/logistic_regression.py
bash run_neural.sh
```
