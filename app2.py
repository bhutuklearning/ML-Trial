# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score

# st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")

# # Load dataset
# df = pd.read_csv("expanded_house_price_with_features.csv")

# # Preprocess binary and categorical features
# df['LivingRoom'] = df['LivingRoom'].map({'Yes': 1, 'No': 0})
# df['Parking'] = df['Parking'].map({'Yes': 1, 'No': 0})
# df = pd.get_dummies(df, columns=['Furnishing'], drop_first=True)

# # Features and target
# X_raw = df.drop(columns=['Price (INR)']).astype(float)
# y = df['Price (INR)'].astype(float).values.reshape(-1, 1)

# # Normalize features
# scaler = StandardScaler()
# X = scaler.fit_transform(X_raw)

# # Initialize weights and bias
# np.random.seed(42)
# weights = np.random.randn(X.shape[1], 1)
# bias = 0.0

# # Hyperparameters
# lr = 1e-2
# epochs = 1000

# # Training with Gradient Descent
# losses = []
# for epoch in range(epochs):
#     y_pred = np.dot(X, weights) + bias
#     error = y_pred - y
#     loss = np.mean(error ** 2)
#     losses.append(loss)

#     dW = (2 / len(X)) * np.dot(X.T, error)
#     dB = (2 / len(X)) * np.sum(error)

#     weights -= lr * dW
#     bias -= lr * dB

# # Final prediction and metrics
# y_final_pred = np.dot(X, weights) + bias
# r2 = r2_score(y, y_final_pred)

# # ---------------- Streamlit Interface ----------------

# st.title("🏠 House Price Prediction App")
# st.markdown("This app uses **multiple linear regression** trained from scratch to predict house prices.")

# # Sidebar Input
# with st.sidebar:
#     st.header("🔧 Input Features")
#     area = st.number_input("Area (sqft)", 200, 10000, 1200)
#     bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)
#     bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
#     livingroom = st.radio("Living Room?", ["Yes", "No"])
#     parking = st.radio("Parking?", ["Yes", "No"])
#     furnishing = st.selectbox("Furnishing", df.columns[df.columns.str.startswith("Furnishing_")].str.replace("Furnishing_", ""))

# # Construct user input vector
# user_input = pd.DataFrame([{
#     "Area (sqft)": area,
#     "Bedrooms": bedrooms,
#     "Bathrooms": bathrooms,
#     "LivingRoom": 1 if livingroom == "Yes" else 0,
#     "Parking": 1 if parking == "Yes" else 0,
#     **{col: int(col.endswith(furnishing)) for col in X_raw.columns if col.startswith("Furnishing_")}
# }])

# # Ensure all features exist in input
# for col in X_raw.columns:
#     if col not in user_input.columns:
#         user_input[col] = 0

# # Reorder and scale
# user_input = user_input[X_raw.columns]
# user_scaled = scaler.transform(user_input)
# predicted_price = float((np.dot(user_scaled, weights) + bias).item())

# # ---------------- Output Section ----------------

# st.subheader("💰 Predicted House Price:")
# st.success(f"₹ {predicted_price:,.2f}")

# st.markdown(f"📈 Model R² Score: `{r2:.4f}`")

# # ---------------- Visualizations ----------------

# # Plot: Loss over Epochs
# st.subheader("📉 Training Loss Curve")
# fig1, ax1 = plt.subplots()
# ax1.plot(losses, color='purple')
# ax1.set_xlabel("Epoch")
# ax1.set_ylabel("MSE Loss")
# ax1.set_title("Loss over Training Epochs")
# ax1.grid(True)
# st.pyplot(fig1)

# # Plot: Feature Importance
# st.subheader("📊 Feature Importance (Weights)")
# fig2, ax2 = plt.subplots()
# ax2.barh(X_raw.columns, weights.flatten(), color='teal')
# ax2.set_xlabel("Weight")
# ax2.set_title("Learned Weights per Feature")
# st.pyplot(fig2)

# # Plot: Predicted vs Actual Prices
# st.subheader("🔍 Predicted vs Actual Prices")
# fig3, ax3 = plt.subplots()
# ax3.scatter(y, y_final_pred, alpha=0.5, color='darkorange', edgecolors='k')
# ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
# ax3.set_xlabel("Actual Price (INR)")
# ax3.set_ylabel("Predicted Price (INR)")
# ax3.set_title("Actual vs Predicted Prices")
# st.pyplot(fig3)

# # Plot: Residuals
# st.subheader("📉 Residuals (Prediction Error)")
# residuals = y_final_pred - y
# fig4, ax4 = plt.subplots()
# ax4.scatter(y_final_pred, residuals, alpha=0.5, color='crimson')
# ax4.axhline(0, color='black', linestyle='--')
# ax4.set_xlabel("Predicted Price")
# ax4.set_ylabel("Residual (Predicted - Actual)")
# ax4.set_title("Residuals Plot")
# st.pyplot(fig4)

# # Plot: Distribution of House Prices
# st.subheader("📊 Distribution of House Prices")
# fig5, ax5 = plt.subplots()
# ax5.hist(y, bins=30, color='skyblue', edgecolor='black')
# ax5.set_title("Distribution of House Prices (INR)")
# ax5.set_xlabel("Price")
# ax5.set_ylabel("Frequency")
# st.pyplot(fig5)

# st.markdown("✅ This model was trained using gradient descent on a normalized dataset.")




import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")

# Load dataset
df = pd.read_csv("expanded_house_price_with_features.csv")

# Preprocess binary and categorical features
df['LivingRoom'] = df['LivingRoom'].map({'Yes': 1, 'No': 0})
df['Parking'] = df['Parking'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, columns=['Furnishing'], drop_first=True)

# Features and target
X_raw = df.drop(columns=['Price (INR)']).astype(float)
y = df['Price (INR)'].astype(float).values.reshape(-1, 1)

# Manual normalization (z-score)
X_mean = X_raw.mean()
X_std = X_raw.std()
X = (X_raw - X_mean) / X_std

# Initialize weights and bias
np.random.seed(42)
weights = np.random.randn(X.shape[1], 1)
bias = 0.0

# Hyperparameters
lr = 1e-2
epochs = 1000

# Training with Gradient Descent
losses = []
for epoch in range(epochs):
    y_pred = np.dot(X, weights) + bias
    error = y_pred - y
    loss = np.mean(error ** 2)
    losses.append(loss)

    dW = (2 / len(X)) * np.dot(X.T, error)
    dB = (2 / len(X)) * np.sum(error)

    weights -= lr * dW
    bias -= lr * dB

# Manual R² score function
def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# Final prediction and metrics
y_final_pred = np.dot(X, weights) + bias
r2 = r2_score_manual(y, y_final_pred)

# ---------------- Streamlit Interface ----------------

st.title("🏠 House Price Prediction App")
st.markdown("This app uses **multiple linear regression** trained from scratch to predict house prices.")

# Sidebar Input
with st.sidebar:
    st.header("🔧 Input Features")
    area = st.number_input("Area (sqft)", 200, 10000, 1200)
    bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
    livingroom = st.radio("Living Room?", ["Yes", "No"])
    parking = st.radio("Parking?", ["Yes", "No"])
    furnishing = st.selectbox("Furnishing", df.columns[df.columns.str.startswith("Furnishing_")].str.replace("Furnishing_", ""))

# Construct user input vector
user_input = pd.DataFrame([{
    "Area (sqft)": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "LivingRoom": 1 if livingroom == "Yes" else 0,
    "Parking": 1 if parking == "Yes" else 0,
    **{col: int(col.endswith(furnishing)) for col in X_raw.columns if col.startswith("Furnishing_")}
}])

# Ensure all features exist in input
for col in X_raw.columns:
    if col not in user_input.columns:
        user_input[col] = 0

# Reorder and normalize manually
user_input = user_input[X_raw.columns]
user_scaled = (user_input - X_mean) / X_std
predicted_price = float((np.dot(user_scaled, weights) + bias).item())

# ---------------- Output Section ----------------

st.subheader("💰 Predicted House Price:")
st.success(f"₹ {predicted_price:,.2f}")

st.markdown(f"📈 Model R² Score: `{r2:.4f}`")

# ---------------- Visualizations ----------------

st.subheader("📉 Training Loss Curve")
fig1, ax1 = plt.subplots()
ax1.plot(losses, color='purple')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax1.set_title("Loss over Training Epochs")
ax1.grid(True)
st.pyplot(fig1)

st.subheader("📊 Feature Importance (Weights)")
fig2, ax2 = plt.subplots()
ax2.barh(X_raw.columns, weights.flatten(), color='teal')
ax2.set_xlabel("Weight")
ax2.set_title("Learned Weights per Feature")
st.pyplot(fig2)

st.subheader("🔍 Predicted vs Actual Prices")
fig3, ax3 = plt.subplots()
ax3.scatter(y, y_final_pred, alpha=0.5, color='darkorange', edgecolors='k')
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax3.set_xlabel("Actual Price (INR)")
ax3.set_ylabel("Predicted Price (INR)")
ax3.set_title("Actual vs Predicted Prices")
st.pyplot(fig3)

st.subheader("📉 Residuals (Prediction Error)")
residuals = y_final_pred - y
fig4, ax4 = plt.subplots()
ax4.scatter(y_final_pred, residuals, alpha=0.5, color='crimson')
ax4.axhline(0, color='black', linestyle='--')
ax4.set_xlabel("Predicted Price")
ax4.set_ylabel("Residual (Predicted - Actual)")
ax4.set_title("Residuals Plot")
st.pyplot(fig4)

st.subheader("📊 Distribution of House Prices")
fig5, ax5 = plt.subplots()
ax5.hist(y, bins=30, color='skyblue', edgecolor='black')
ax5.set_title("Distribution of House Prices (INR)")
ax5.set_xlabel("Price")
ax5.set_ylabel("Frequency")
st.pyplot(fig5)

st.markdown("✅ This model was trained using gradient descent on a normalized dataset.")
