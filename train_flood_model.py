import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. Load and Prepare Data ---
# For this example, we'll create a sample DataFrame.
# In a real scenario, you would load your CSVs here.
data = {
    'date': pd.to_datetime(pd.date_range(start='1/1/2020', periods=1000)),
    'rainfall_mm': np.random.uniform(0, 50, 1000),
    'ndvi': np.random.uniform(0.1, 0.9, 1000),
    'elevation_m': np.random.uniform(10, 500, 1000),
    'flood_occurred': np.random.randint(0, 2, 1000) # Target variable: 1 for flood, 0 for no flood
}
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

print("--- Sample Data Head ---")
print(df.head())

# --- 2. Feature Scaling ---
scaler = MinMaxScaler()
df[['rainfall_mm', 'ndvi', 'elevation_m']] = scaler.fit_transform(df[['rainfall_mm', 'ndvi', 'elevation_m']])

# --- 3. Create Time-Series Sequences for LSTM ---
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data.iloc[i:(i + sequence_length), :-1]
        y = data.iloc[i + sequence_length, -1]
        xs.append(x.values)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 10 # Use 10 days of data to predict the 11th day
X, y = create_sequences(df, sequence_length)

# --- 4. Split Data for Training and Testing ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Data Shapes ---")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# --- 5. Build and Train the LSTM Model ---
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification (flood or no flood)
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Training LSTM Model ---")
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# --- 6. Get LSTM Predictions (Features for XGBoost) ---
lstm_train_pred = lstm_model.predict(X_train)
lstm_test_pred = lstm_model.predict(X_test)

# --- 7. Prepare Data for XGBoost ---
# We need to reshape the original data to be 2D
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Add the LSTM predictions as a new feature
X_train_hybrid = np.hstack((X_train_flat, lstm_train_pred))
X_test_hybrid = np.hstack((X_test_flat, lstm_test_pred))

# --- 8. Build and Train the XGBoost Model ---
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

print("\n--- Training XGBoost Model ---")
xgb_model.fit(X_train_hybrid, y_train)

# --- 9. Evaluate the Hybrid Model ---
accuracy = xgb_model.score(X_test_hybrid, y_test)
print(f"\nHybrid LSTM+XGBoost Model Accuracy: {accuracy * 100:.2f}%")

# --- 10. Save the Models ---
lstm_model.save('flood_lstm_model.h5')
xgb_model.save_model('flood_xgb_model.json')

print("\nModels saved successfully!")