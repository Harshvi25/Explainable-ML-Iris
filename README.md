# Explainable-ML-Iris

📌 Overview
This project demonstrates the use of Explainable AI (XAI) techniques to interpret the predictions of a Neural Network trained on the Iris dataset. Using LIME (Local Interpretable Model-Agnostic Explanations), we analyze how the model makes decisions and ensure transparency in its predictions.

🔍 Why XAI?
As AI models become more complex, understanding their decision-making process is crucial for trust, fairness, and accountability. XAI techniques like LIME help in:
  *  Making AI decisions interpretable
  *  Identifying potential biases
  *  Enhancing trust and reliability in AI systems

🛠 Technologies Used
  *  Python
  *  TensorFlow/Keras (for building the neural network)
  *  Scikit-learn (for dataset handling)
  *  LIME (for explainability)
  *  Matplotlib (for visualization)

📊 Project Workflow
  1.Load the Iris Dataset: A classic dataset with three flower species based on sepal/petal dimensions.
  2.Preprocess the Data: One-hot encode labels and split into training/testing sets.
  3.Build a Neural Network: A simple feedforward neural network with two layers.
  4.Train the Model: Using categorical cross-entropy and Adam optimizer.
  5.Apply LIME for Explainability:
      *  Generate explanations for predictions.
      *  Visualize feature importance using LIME.

