import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
 
# =========================================================================
# 0. Report Output Directory (Real Images and Tables)
# =========================================================================
output_dir = "reloop_report_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
# =========================================================================
# 1. Data Configurations and Paths
# =========================================================================
DATA_PATH = "dataset_dual_hands"  # New folder containing 126 keypoints
actions = np.array(sorted(os.listdir(DATA_PATH)))  # Includes (Nothing, Hello, All Done, ...)
label_map = {label: num for num, label in enumerate(actions)}
 
sequences, labels = [], []
 
# Real counter for actual samples per class - used later for plotting 
# the true, data-driven "Dataset Distribution"
samples_per_class_real = {action: 0 for action in actions}
 
print(f"Loading data for classes: {actions}")
 
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    no_seqs = len([d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))])
 
    for sequence in range(no_seqs):
        window = []
        for frame_num in range(20):  # Sequence length is 20 frames as defined in data collection
            try:
                res = np.load(os.path.join(action_path, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            except:
                continue
 
        if len(window) == 20:
            sequences.append(window)
            labels.append(label_map[action])
            samples_per_class_real[action] += 1
 
# Convert data to Numpy arrays
X = np.array(sequences)  # Expected shape: (num_sequences, 20, 126)
y = to_categorical(labels).astype(int)
 
# Split data (80% training and 20% testing for rigorous evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# =========================================================================
# 2. Model Architecture
# =========================================================================
model = Sequential()
 
# First Layer: LSTM handling 126 input keypoints
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
 
# Output Layer: Softmax giving class probabilities for each word (including Nothing)
model.add(Dense(actions.shape[0], activation='softmax'))
 
# =========================================================================
# 3. Training Configurations
# =========================================================================
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)
 
print("Starting training process...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)
 
# =========================================================================
# 4. Save Model
# =========================================================================
model.save('asl_dual_hands_model.keras')
print("Model trained and saved successfully!")
 
# =========================================================================
# 5. Real Evaluation: True Model Predictions on Actual Test Data (X_test)
#    (No generated or synthetic data - derived purely from model outputs)
# =========================================================================
print("\nCalculating true evaluation metrics from the trained model...")
 
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
 
# Map class indices back to original string action labels for plots and tables
y_true_labels = actions[y_true_classes]
y_pred_labels = actions[y_pred_classes]
 
# -------------------------------------------------------------------
# ASSET 1: Confusion Matrix (Generated from true model predictions)
# -------------------------------------------------------------------
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=actions)
 
plt.figure(figsize=(16, 13))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=actions, yticklabels=actions,
            annot_kws={"size": 8})
plt.title('ReLoop Sign Language Recognition System: Confusion Matrix (Real Test Results)',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual ASL Sign', fontsize=12, fontweight='bold')
plt.xlabel('Predicted ASL Sign', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_confusion_matrix.png'), dpi=300)
plt.close()
 
# -------------------------------------------------------------------
# ASSET 2: Learning Curves (Derived directly from history object)
# -------------------------------------------------------------------
real_epochs = range(1, len(history.history['accuracy']) + 1)
 
plt.figure(figsize=(15, 6))
 
plt.subplot(1, 2, 1)
plt.plot(real_epochs, history.history['accuracy'], 'b-o', markersize=4, label='Training Accuracy')
plt.plot(real_epochs, history.history['val_accuracy'], 'r-s', markersize=4, label='Validation Accuracy')
plt.title('Model Learning Evaluation: Accuracy Curves', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy Score', fontsize=10)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
 
plt.subplot(1, 2, 2)
plt.plot(real_epochs, history.history['loss'], 'b-o', markersize=4, label='Training Loss')
plt.plot(real_epochs, history.history['val_loss'], 'r-s', markersize=4, label='Validation Loss')
plt.title('Model Learning Evaluation: Loss Curves', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss Value', fontsize=10)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
 
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_learning_curves.png'), dpi=300)
plt.close()
 
# -------------------------------------------------------------------
# ASSET 3: Per-Class Precision/Recall/F1 (From classification report)
# -------------------------------------------------------------------
report_dict = classification_report(y_true_labels, y_pred_labels,
                                     labels=actions, target_names=actions,
                                     output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose().loc[actions]
 
plt.figure(figsize=(16, 7))
x = np.arange(len(actions))
width = 0.25
 
plt.bar(x - width, report_df['precision'], width, label='Precision', color='#1f77b4')
plt.bar(x, report_df['recall'], width, label='Recall', color='#aec7e8')
plt.bar(x + width, report_df['f1-score'], width, label='F1-Score', color='#ff7f0e')
 
plt.title('Evaluation Metrics Analysis Across All Dictionary Words (Real Results)',
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('ASL Dictionary Words', fontsize=11, fontweight='bold')
plt.ylabel('Score (0.0 - 1.0)', fontsize=11, fontweight='bold')
plt.xticks(x, actions, rotation=45, ha='right', fontsize=9)
plt.ylim(0, 1.1)
plt.legend(loc='lower left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_metrics_by_class.png'), dpi=300)
plt.close()
 
# -------------------------------------------------------------------
# ASSET 4: Dataset Class Distribution (Counted during data loading)
# -------------------------------------------------------------------
class_counts_real = [samples_per_class_real[action] for action in actions]
 
plt.figure(figsize=(15, 5))
sns.barplot(x=list(actions), y=class_counts_real, palette="Blues_d")
plt.title('Dataset Analysis: Actual Sample Distribution Per Class', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('ASL Sign Classes', fontsize=11, fontweight='bold')
plt.ylabel('Total Sequences Loaded', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_dataset_distribution.png'), dpi=300)
plt.close()
 
# -------------------------------------------------------------------
# ASSET 5: Complete Evaluation Table - CSV Export for Reports
# -------------------------------------------------------------------
full_report_df = pd.DataFrame(report_dict).transpose()
full_report_df.round(2).to_csv(os.path.join(output_dir, '5_classification_report_table.csv'))
 
print("\n" + "=" * 70)
print("🎯 SUCCESS: 5 Report Assets Generated From REAL Model Results!")
print(f"1. Confusion Matrix          -> {output_dir}/1_confusion_matrix.png")
print(f"2. Learning Curves           -> {output_dir}/2_learning_curves.png")
print(f"3. Per-Class Metrics Chart   -> {output_dir}/3_metrics_by_class.png")
print(f"4. Dataset Distribution      -> {output_dir}/4_dataset_distribution.png")
print(f"5. Classification Report CSV -> {output_dir}/5_classification_report_table.csv")
print("=" * 70)