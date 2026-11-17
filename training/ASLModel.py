import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Data configuration and paths
DATA_PATH = "dataset_dual_hands" # New folder containing 126 keypoints
actions = np.array(sorted(os.listdir(DATA_PATH))) # Includes (Nothing, Hello, All Done, ...)
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

print(f"Loading data for classes: {actions}")

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    # Check the actual number of video sequences in the folder
    no_seqs = len([d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))])
    
    for sequence in range(no_seqs):
        window = []
        for frame_num in range(20): # Sequence length is 20 frames
            try:
                res = np.load(os.path.join(action_path, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            except:
                continue
        
        if len(window) == 20:
            sequences.append(window)
            labels.append(label_map[action])

# Convert data to Numpy arrays
X = np.array(sequences) # Expected shape: (num_sequences, 20, 126)
y = to_categorical(labels).astype(int)

# Split data (80% training and 20% testing for proper validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model Architecture
model = Sequential()

# First Layer: LSTM handling 126 keypoints
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(20, 126)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Second Layer: Deeper LSTM to extract complex movement patterns
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Fully Connected Layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Output Layer: Softmax predicting probabilities for each class (including Nothing)
model.add(Dense(actions.shape[0], activation='softmax'))

# 3. Training Configurations
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training optimizations:
# - Early stopping to prevent overfitting if validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
# - Reduce learning rate when validation loss plateaus to fine-tune weights
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)

print("Starting training process...")
history = model.fit(
    X_train, y_train, 
    epochs=200, 
    batch_size=16, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)

# 4. Save Model
model.save('asl_dual_hands_model.keras')
print("Model trained and saved successfully!")