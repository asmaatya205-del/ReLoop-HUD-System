import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
 
# =========================================================================
# 0. Report Output Directory (Shared with ASL report folder to unify assets)
# =========================================================================
output_dir = "emotion_report_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
# =========================================================================
# 1. Data Loading and Preprocessing
# =========================================================================
label_map = {"Happy": 0, "Neutral": 1, "Sad": 2}
actions = np.array(['Happy', 'Neutral', 'Sad'])
 
X, y = [], []
samples_per_class_real = {action: 0 for action in actions}
 
print("--- Loading Data ---")
for action in actions:
    path = os.path.join('Emotion_Data', action)
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
 
    for file in files:
        res = np.load(os.path.join(path, file))
        X.append(res)
        y.append(label_map[action])
    samples_per_class_real[action] = len(files)
    print(f"Loaded {len(files)} samples for: {action}")
 
X = np.array(X)
y_int = np.array(y)  # Keep a copy of original integer labels for stratified train_test_split
y = to_categorical(y_int).astype(int)
 
# Split data (80% training, 20% testing) with stratification to preserve class distributions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_int
)
 
# =========================================================================
# 2. Model Architecture
# =========================================================================
emotion_model = Sequential()
 
emotion_model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.3))
 
emotion_model.add(Dense(128, activation='relu'))
emotion_model.add(Dropout(0.3))
 
emotion_model.add(Dense(64, activation='relu'))
 
emotion_model.add(Dense(actions.shape[0], activation='softmax'))
 
# =========================================================================
# 3. Training Configurations
# =========================================================================
emotion_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
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
 
# =========================================================================
# 4. Save Model
# =========================================================================
emotion_model.save('emotion_model.keras')
print("\n--- Success! Model saved as: emotion_model.keras ---")
 
loss, accuracy = emotion_model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
 
# =========================================================================
# 5. Real Evaluation: True Model Predictions on Actual Test Data (X_test)
# =========================================================================
print("\nCalculating true evaluation metrics from the trained emotion model...")
 
y_pred_probs = emotion_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
 
y_true_labels = actions[y_true_classes]
y_pred_labels = actions[y_pred_classes]
 
# -------------------------------------------------------------------
# ASSET 1: Confusion Matrix (Real 3x3 Grid)
# -------------------------------------------------------------------
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=actions)
 
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=True,
            xticklabels=actions, yticklabels=actions)
plt.title('ReLoop Emotion Recognition: Confusion Matrix (Real Test Results)',
          fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Actual Emotion', fontsize=10, fontweight='bold')
plt.xlabel('Predicted Emotion', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '6_emotion_confusion_matrix.png'), dpi=300)
plt.close()
 
# -------------------------------------------------------------------
# ASSET 2: Learning Curves (Extracted from history.history)
# -------------------------------------------------------------------
real_epochs = range(1, len(history.history['accuracy']) + 1)
 
plt.figure(figsize=(12, 5))
 
plt.subplot(1, 2, 1)
plt.plot(real_epochs, history.history['accuracy'], 'b-o', markersize=4, label='Training Accuracy')
plt.plot(real_epochs, history.history['val_accuracy'], 'r-s', markersize=4, label='Validation Accuracy')
plt.title('Emotion Model: Accuracy Curves', fontsize=11, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
 
plt.subplot(1, 2, 2)
plt.plot(real_epochs, history.history['loss'], 'b-o', markersize=4, label='Training Loss')
plt.plot(real_epochs, history.history['val_loss'], 'r-s', markersize=4, label='Validation Loss')
plt.title('Emotion Model: Loss Curves', fontsize=11, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
 
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '7_emotion_learning_curves.png'), dpi=300)
plt.close()
 
# -------------------------------------------------------------------
# ASSET 3: Per-Class Precision/Recall/F1 (True Evaluation)
# -------------------------------------------------------------------
report_dict = classification_report(y_true_labels, y_pred_labels,
                                     labels=actions, target_names=actions,
                                     output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose().loc[actions]
 
plt.figure(figsize=(8, 5))
x = np.arange(len(actions))
width = 0.25
 
plt.bar(x - width, report_df['precision'], width, label='Precision', color='#6a3d9a')
plt.bar(x, report_df['recall'], width, label='Recall', color='#cab2d6')
plt.bar(x + width, report_df['f1-score'], width, label='F1-Score', color='#ff7f0e')
 
plt.title('Emotion Model: Per-Class Evaluation Metrics (Real Results)',
          fontsize=12, fontweight='bold', pad=15)
plt.xticks(x, actions)
plt.ylim(0, 1.1)
plt.legend(loc='lower left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '8_emotion_metrics_by_class.png'), dpi=300)
plt.close()
 
# -------------------------------------------------------------------
# ASSET 4: Dataset Class Distribution (True Loading Counts)
# -------------------------------------------------------------------
class_counts_real = [samples_per_class_real[action] for action in actions]
 
plt.figure(figsize=(6, 4))
sns.barplot(x=list(actions), y=class_counts_real, palette="Purples_d")
plt.title('Emotion Dataset: Actual Sample Distribution', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Total Samples Loaded')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '9_emotion_dataset_distribution.png'), dpi=300)
plt.close()
 
# -------------------------------------------------------------------
# ASSET 5: Complete Evaluation Table - CSV Export
# -------------------------------------------------------------------
full_report_df = pd.DataFrame(report_dict).transpose()
full_report_df.round(2).to_csv(os.path.join(output_dir, '10_emotion_classification_report.csv'))
 
print("\n" + "=" * 70)
print("🎯 SUCCESS: Emotion Model Report Assets Generated From REAL Results!")
print(f"6.  Confusion Matrix          -> {output_dir}/6_emotion_confusion_matrix.png")
print(f"7.  Learning Curves           -> {output_dir}/7_emotion_learning_curves.png")
print(f"8.  Per-Class Metrics Chart   -> {output_dir}/8_emotion_metrics_by_class.png")
print(f"9.  Dataset Distribution      -> {output_dir}/9_emotion_dataset_distribution.png")
print(f"10. Classification Report CSV -> {output_dir}/10_emotion_classification_report.csv")
print("=" * 70)