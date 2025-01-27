<h1>Image Classification with TensorFlow</h1>

<p>This repository contains code for building an image classification model using TensorFlow and Keras. The model is trained to classify images of fruits and vegetables into different categories. </p>

<h2>Dataset</h2>
<p>The dataset used for training, validation, and testing contains images of various fruits and vegetables. It is organized into three subsets:</p>
<UL>
  <LI>Training Data: Used to train the model</LI>
  <li>Validation Data: Used to evaluate the model's performance during training and tune hyperparameters</li>
  <LI>Test Data: Used to assess the model's performance on unseen data</LI>
</UL>

<h2>Model Architecture</h2>

<p>The model architecture consists of a convolutional neural network (CNN) implemented using TensorFlow's Sequential API. The architecture includes several convolutional layers followed by max-pooling layers for feature extraction. The flattened output is then fed into fully connected layers for classification. Dropout regularization is applied to prevent overfitting, and the final layer uses a softmax activation function to output class probabilities.</p>

<h2>Training</h2>
<p>The model is trained using the training data with an Adam optimizer and sparse categorical cross-entropy loss function. Training is performed for a specified number of epochs, and the model's performance is monitored using the validation data.
</p>

<h2>Evaluation</h2>
<p>After training, the model's performance is evaluated using the test data to assess its accuracy in classifying new, unseen images.
</p>

<h2>Prediction</h2>
<p>The trained model can be used to make predictions on new images. Simply provide the path to the image file, and the model will output the predicted class along with the corresponding confidence score.</p>

<h2>Usage</h3>
<p>To train the model, run the provided Python script. Make sure to adjust the paths to the dataset directories and customize the model architecture and hyperparameters as needed. After training, you can save the trained model for future use.</p>

<h2>Requirements</h2>
<ul>
  <li>TensorFlow</li>
  <li>NumPy</li>
  <li>Matplotlb</li>
</ul>
