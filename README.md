# Hand-Written-Digit-Classifier-Using-Pytorch-CNN-
Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model is trained to recognize digits (0-9) with high accuracy and can predict handwritten numbers.

Features

Uses a CNN architecture with two convolutional layers, ReLU activations, and max pooling.

Implements dropout to reduce overfitting.

Trained using CrossEntropyLoss and optimized with Adam.

Achieves high accuracy on the MNIST test dataset.

Installation

Ensure you have Python and the required dependencies installed.

1️⃣ Clone the Repository

git clone https://github.com/your-username/mnist-pytorch.git
cd mnist-pytorch

2️⃣ Install Dependencies

pip install torch torchvision matplotlib

Usage

Train the Model

Run the following command to train the model:

python train.py

Test the Model

Once training is complete, you can evaluate the model:

python test.py

Predict on a Sample Image

To test the model on a random MNIST image, run:

python predict.py

Model Architecture

The CNN model consists of the following layers:

Conv2d(1, 32, 3) → ReLU → MaxPool

Conv2d(32, 64, 3) → ReLU → MaxPool

Fully Connected Layer (128 neurons) → ReLU → Dropout

Output Layer (10 neurons for digits 0-9)

Results

After training for 5 epochs, the model achieves an accuracy of 98%+ on the MNIST test dataset.

Contributing

Feel free to open an issue or submit a pull request if you'd like to improve this project.

License

This project is open-source and available under the MIT License.

Acknowledgments

PyTorch (https://pytorch.org)

MNIST Dataset (http://yann.lecun.com/exdb/mnist/)

