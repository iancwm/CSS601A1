# CS601 Introduction to Artificial Intelligence - Assignment 1
Assignment 1 covers Bayesian networks and image recognition using `keras`

## Question 1 - Directed Bayesian Network
Determining if variables are *d-separated* and therefore conditionally independent from one another.

## Question 2 - Probability
Determine if three random variables are independent, and accordingly compute the probabilities of a given outcome, as well as the MAP.

## Question 3 - BayesNet construction
Question 3 is done using `pgmpy`, and the code can be found and executed in the notebook. It constructs a network using the relationships given, and queries for required probabilities using variable elimination.

## Question 4 - Fashion MNIST
Trains a vanilla ANN on a reduced Fashion MNIST dataset. Uses an `Adagrad` optimizer and categorical cross entropy loss function. `ReduceLROnPlateau` is added as a callback to allow for continued improvements after model metrics stagnate.

The model achieved 89% accuracy on the train set and 86% on the evaluation set.

## Question 5 - CIFAR10 data set. 
Trains a CNN on the `CIFAR10` data set. Uses an `rmsprop` optimizer and sparse categorical cross entropy loss function. `ReduceLROnPlateau` is added as a callback to allow for continued improvements after model metrics stagnate.

The base model used is `MobileNetV2`. I kept the original weights as required by the question, and added additional convolutional layers.

The model achieved a 95.5% accuracy on the train set but only a 78% accuracy on the evaluation set, suggesting overfitting.

While not documented, it was observed that when I chose to retrain all layers including the original `MobileNetV2` layer, the resulting model could not differentiate vehicles as well, mistaking trucks for cars and vice versa more frequently. This suggests Google's training set for MobileNet included a lot of self driving car footage, since they are well known for it.

*NOTE:* The code to download `CIFAR10` data and train an extended `MobileNetV2` model using transfer learning will create a folder after running to save the model. The model was trained on Google Colab to take advantage of TPU acceleration, with the notebook uploaded for viewing.