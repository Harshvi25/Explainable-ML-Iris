import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import lime
from lime.lime_tabular import LimeTabularExplainer          
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data  
y = iris.target  

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output = False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1)) # It'll reshape y into 2D array

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = keras.Sequential([keras.layers.Input(shape = (4,)),
        keras.layers.Dense(10, activation = 'relu'),
        keras.layers.Dense(3, activation = 'softmax')])

# Compile & Train the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=2)

# Create the LIME explainer
explainer = LimeTabularExplainer(X_train,training_labels=y_train,mode="classification")

# Explain the model's prediction for the first test instance
explanation = explainer.explain_instance(X_test[0], model.predict)

# Visualize the explanation
explanation.as_pyplot_figure()
plt.show()

# Explain the model's prediction for the first test instance
explanation = explainer.explain_instance(X_test[0], model.predict)

# Visualize the explanation
explanation.as_pyplot_figure()  # the chart updates each time due to randomness
plt.show()













