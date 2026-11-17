import cv2
import mediapipe as mp
import numpy as np
import os

# --- Configuration ---
word = "War"  # Start by recording 'Nothing' then move to other actions
no_sequences = 60 
sequence_length = 20 
data_path = os.path.join('dataset_dual_hands', word)

os.makedirs(data_path, exist_ok=True)
for seq in range(no_sequences):
    os.makedirs(os.path.join(data_path, str(seq)), exist_ok=True)

mp_hands = mp.solutions.hands
# Set maximum number of hands to 2 for dual hand detection
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

for seq in range(no_sequences):
    print(f"Recording {word} - Sequence {seq}")
    for t in range(3, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, f'PREPARE {word} in {t}...', (150,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow('Collection Feed', frame)
        cv2.waitKey(400)

    for frame_num in range(sequence_length):
        ret, frame = cap.read()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        # Initialize two arrays for both hands (63 keypoints each)
        lh_landmarks = np.zeros(21 * 3)
        rh_landmarks = np.zeros(21 * 3)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine hand handedness (Left or Right)
                label = results.multi_handedness[i].classification[0].label
                
                # Extract keypoint coordinates
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                
                if label == 'Left':
                    lh_landmarks = coords
                else:
                    rh_landmarks = coords

        # Concatenate both hands together (Total of 126 keypoints)
        full_data = np.concatenate([lh_landmarks, rh_landmarks])
        np.save(os.path.join(data_path, str(seq), f"{frame_num}.npy"), full_data)

        cv2.putText(frame, f'RECORDING: {word} Video {seq}', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow('Collection Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()