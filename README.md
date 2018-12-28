# MNIST_Classification
As a introduction to TensorFlow for CSCI 5525 - Machine Learning, we had to carry out digit classification using various techniques on the MNIST dataset. This was used to demonstrate the increased performance of deep learning techniques such as convolutional neural networks (CNN) and recurrent neural networks (RNN) over the standard logistic regression model.

The different model used in thsi comparative analysis are:
1. One Layer Logistic Regression Model (`TFLogReg.py`)
2. Logistic Regression Model with 2 Hidden Layers (`ImproveLogReg.py`)
3. Convolutional Neural Network (`ConvNetTemplate.py`)
4. Long Short-Term Memory (`LSTM.py`)

## Logistic Regression
In this model, the MNIST images were unrolled into a 1D tensor that was fed into the input layer of the model. The input feature values were then directly mapped to the output layer for classification. 

## Improved Logistic Regression
In this model, the simple logistic regression model is improved by adding in two hidden layers of size 256 and 64 before being mapped to the output classes (0-9). This allows the model to learn more complex relationships between the pixel values while condensing the data into fewer nodes at each level. 

## Convolutional Neural Network (CNN)
The model contains two convolution layers using kernels of size 5 x 5 and depth of 32 and 64 respectively. Each convolution layer is followed by a max pool layer of using kernel size of 2 x 2 to decrease the size of the image while increasing the depth. The final image is unrolled into a 1D tensor, which is fed into a fully connected layer before being classified.

## Long Short-Term Memory (LSTM)
The image data is reshaped into a 1D tensor before being fed into an LSTM cell of size 256. The output of the cell is then projected into the output classes to classify the data. 

## Summary

| Left-aligned | Training Time  |    Accuracy    |
| :---         |     :---:      |     :---:      |
| Logistic Regression  |   7.441 s   |   89.67%  |
| Multi-Layer Logistic Regression     |    22.347 s    |    97.28%   |
| CNN     |    210.198 s    |    92.38%   |
| LSTM     |    2243.486 s    |    98.04%   |

## Running the Files
1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/murak038/tv-script-generation.git
cd tv-script-generation
```
2. Run the file from the Terminal
```	
python TFLogReg.py
```
