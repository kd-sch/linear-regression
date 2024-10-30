import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/Nairobi Office Price Ex.csv'
data = pd.read_csv("Nairobi Office Price Ex.csv")

# Extracting the SIZE and PRICE columns for regression
x_data = data['SIZE'].values
y_data = data['PRICE'].values

# Define Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define Gradient Descent function for linear regression
def gradient_descent(x, y, m, c, learning_rate):
    n = len(x)
    y_pred = m * x + c
    # Calculating gradients
    m_gradient = (-2 / n) * np.sum(x * (y - y_pred))
    c_gradient = (-2 / n) * np.sum(y - y_pred)
    # Update weights
    m = m - learning_rate * m_gradient
    c = c - learning_rate * c_gradient
    return m, c

# Initialize random values for slope (m) and intercept (c)

m = np.random.rand()
c = np.random.rand()

# Set hyperparameters
learning_rate = 0.0001  # Smaller learning rate for better convergence
epochs = 10

# Training loop
for epoch in range(epochs):
    # Predict using current m and c
    y_pred = m * x_data + c
    # Compute MSE
    error = mean_squared_error(y_data, y_pred)
    print(f"Epoch {epoch+1}, Mean Squared Error: {error}")
    # Update m and c using gradient descent
    m, c = gradient_descent(x_data, y_data, m, c, learning_rate)

# Plotting the line of best fit after final epoch
plt.scatter(x_data, y_data, color='blue', label="Data Points")
plt.plot(x_data, m * x_data + c, color='red', label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.legend()
plt.show()

# Prediction for office size of 100 sq. ft.
office_size = 100
predicted_price = m * office_size + c
print(f"Predicted office price for 100 sq. ft.: {predicted_price}")
