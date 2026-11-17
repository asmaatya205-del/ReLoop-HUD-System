import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# 1. Data Loading and Preprocessing
# ==========================================================
# Ensure folders exist inside 'Emotion_Data' with these exact names
label_map = {"Happy": 0, "Neutral": 1, "Sad": 2}
actions = np.array(['Happy', 'Neutral', 'Sad'])

X, y = [], []

print("--- Loading Data ---")
for action in actions:
    path = os.path.join('Emotion_Data', action)
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
    
    for file in files:
        res = np.load(os.path.join(path, file))
        X.append(res)
        y.append(label_map[action])
    print(f"Loaded {len(files)} samples for: {action}")

X = np.array(X)
y = to_categorical(y).astype(int)

# Split data (80% training, 20% testing for validation accuracy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================================
# 2. Model Architecture
# ==========================================================
emotion_model = Sequential()

# First Layer: Handle 1404 inputs with BatchNormalization for training stability
emotion_model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
emotion_model.add(BatchNormalization()) 
emotion_model.add(Dropout(0.3))

# Second Layer: Represent complex relationships between facial landmarks
emotion_model.add(Dense(128, activation='relu'))
emotion_model.add(Dropout(0.3))

# Third Layer: Consolidate representations before final classification
emotion_model.add(Dense(64, activation='relu'))

# Output Layer: Softmax to output probability distribution over the 3 emotion classes
emotion_model.add(Dense(actions.shape[0], activation='softmax'))

# ==========================================================
# 3. Compilation & Callbacks
# ==========================================================
emotion_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping: Stop training automatically if validation loss does not improve for 15 consecutive epochs
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

print("\n--- Starting Training ---")
history = emotion_model.fit(
    X_train, y_train, 
    epochs=150, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    callbacks=[early_stop]
)

# ==========================================================
# 4. Model and Results Saving
# ==========================================================
# Saved in .keras format as it is the standard and most stable format in current TensorFlow
emotion_model.save('emotion_model.keras')
print("\n--- Success! ---")
print("Model saved as: emotion_model.keras")

# Evaluate final performance on test data
loss, accuracy = emotion_model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")