import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Load data
df = pd.read_csv("expanded_house_price_dataset.csv")
X = df['Area (sqft)'].values.reshape(-1, 1)
y = df['Price (INR)'].values.reshape(-1, 1)

# Hyperparameters
lr = 1e-8
epochs = 1000

# Initialize weights
weight = np.random.randn()
bias = np.random.randn()

# Training loop
losses, weights, biases = [], [], []
for _ in range(epochs):
    y_pred = weight * X + bias
    error = y_pred - y
    loss = np.mean(error ** 2)
    losses.append(loss)
    weights.append(weight)
    biases.append(bias)
    
    dW = np.mean(2 * X * error)
    dB = np.mean(2 * error)
    
    weight -= lr * dW
    bias -= lr * dB

# Final predictions and evaluation
y_final_pred = weight * X + bias
final_loss = np.mean((y_final_pred - y) ** 2)
r2 = r2_score(y, y_final_pred)

# 🧠 Sidebar for Prediction
st.sidebar.header("🏠 Predict House Price")
user_input = st.sidebar.number_input("Enter Area (in sqft)", min_value=100, max_value=20000, step=100, value=1000)
predicted_price = weight * user_input + bias
st.sidebar.markdown(f"### 💰 Predicted Price: ₹{predicted_price:,.2f}")

# 🧾 Model Info
with st.expander("📊 Model Summary"):
    st.write(f"**Initial Weight & Bias:** {weights[0]:.4f}, {biases[0]:.4f}")
    st.write(f"**Final Weight & Bias:** {weight:.4f}, {bias:.4f}")
    st.write(f"**Initial Loss:** {losses[0]:.2f}")
    st.write(f"**Final Loss:** {final_loss:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

# 📉 Plot 1: Loss over epochs
st.subheader("📉 Loss over Epochs")
fig1, ax1 = plt.subplots()
ax1.plot(losses, color='purple')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("MSE Loss")
ax1.set_title("Loss over Epochs")
ax1.grid(True)
st.pyplot(fig1)

# 📈 Plot 2: Linear Regression Line
st.subheader("📈 Regression Line on Data")
fig2, ax2 = plt.subplots()
ax2.scatter(X, y, label="Actual Data", alpha=0.5)
ax2.plot(X, y_final_pred, color='red', label="Regression Line")
ax2.set_xlabel("Area (sqft)")
ax2.set_ylabel("Price (INR)")
ax2.set_title("House Price vs Area")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# 📊 Plot 3: Weight and Bias over Epochs
st.subheader("📊 Weight and Bias Trajectory")
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
ax3.plot(weights, color='green')
ax3.set_title("Weight over Epochs")
ax3.set_xlabel("Epoch")
ax3.grid(True)

ax4.plot(biases, color='orange')
ax4.set_title("Bias over Epochs")
ax4.set_xlabel("Epoch")
ax4.grid(True)

st.pyplot(fig3)

# 🌐 Plot 4: 3D Loss Surface
st.subheader("🌐 3D Loss Surface with Gradient Descent Path")

weight_range = np.linspace(min(weights)-500, max(weights)+500, 50)
bias_range = np.linspace(min(biases)-100000, max(biases)+100000, 50)
W, B = np.meshgrid(weight_range, bias_range)
Loss_surface = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        pred_grid = W[i, j] * X + B[i, j]
        Loss_surface[i, j] = np.mean((pred_grid - y) ** 2)

fig4 = plt.figure(figsize=(10, 6))
ax = fig4.add_subplot(111, projection='3d')
ax.plot_surface(W, B, Loss_surface, cmap='viridis', alpha=0.6)
ax.plot(weights, biases, losses, color='red', label='GD Path', linewidth=2)
ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Loss")
ax.set_title("3D Loss Surface and Gradient Descent Path")
st.pyplot(fig4)
