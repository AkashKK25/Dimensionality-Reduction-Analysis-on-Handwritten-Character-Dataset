# Project 2 - Dimensionality Reduction


## Dataset - Custom Handwritten Characters

The datasets used in this project can be found in Canvas - [download them here](https://ufl.instructure.com/courses/479519/files/folder/Project%202).

## File structure

This repository contains the following files:   
1. "Dimensionality Reduction - Answers.ipynb"
  - This python notebook file contains all the question statements and answers provided with code implementations and visualizations that were also included in the report.
  - This file only contains the answers with the training set.
2. "Dimensionality Reduction - Answers.pdf"
  - This is a pdf version of the previous file for better readability in GitHub.
3. "training.ipynb"
  - This file contains streamlined code for all the trainings and experimentations done on the training set.
  - This code also saves the trained models in pickle files.
4. "test.ipynb"
  - This python notebook contains the code to test on the test data.
  - To run this file trained model files (pickle files stored in training.ipynb) are required.
5. "Project 2 Report.pdf"
  - The 4-page IEEE Report in the PDF format for this project.
6. "t_train.npy"
  - The corrected labels for the training data stored as numpy arrays used in training the models.
7. "t_test.npy"
  - The corrected labels for the test data stored as numpy arrays used in testing the models.

## How to use the code

1. Have the datasets in the same place as you have the training.ipynb and test.ipynb files.
2. Run all the cells in training.ipynb. This will train the models on the training data and create trained model files as pickle (.pkl) files in the same file location.
3. Make sure that the pickle files are present in the same location and run all the cells in the test.ipynb to test the models.

## To run this code on a different test set

To run this code on your own test set, first you need to convert your test data into grayscale images and then to numpy arrays. If you have an image to test on, you can use the following code snippet to do that:

```
import cv2
import numpy as np
import joblib

class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', '$', '#']

image = cv2.imread('Image/location.jpeg', cv2.IMREAD_GRAYSCALE) # replace the image location

image = np.array([cv2.resize(image.reshape(D,D),(50,50)).reshape(2500)]) # replace D with your image dimensions

model = joblib.load('Trained_Model.pkl') # replace the model name

pred = model.predict(image) #If you want to predict. Use fit_transform or transform for visualization instead

print("Model's prediction:", class_names[int(pred)])
```

## Author
Akash Kumar Kondaparthi