import cv2
import mediapipe as mp
import numpy as np
import os
import time

# 1. Configuration
EMOTION = "Neutral" # Change the name for each emotion (Happy, Sad, Neutral)
DATA_PATH = os.path.join('Emotion_Data', EMOTION)
if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH)

# 2. Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1,
    refine_landmarks=False # Disabled for optimization and to prevent lag
)

cap = cv2.VideoCapture(0)

print(f"--- Pro Collection Mode: {EMOTION} ---")
print("Instructions: Move your head slightly and change expression intensity.")
print("Press 'S' to Start, 'Q' to Quit")

collecting = False
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Image conversion
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        
        # Draw mesh for positioning alignment
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
        
        if collecting and count < 500:
            # --- Advanced Processing ---
            # 1. Use the nose tip as a reference point for normalization
            nose = lms[1]
            
            # 2. Extract and scale landmarks to amplify signal strength for the model
            # Multiplying values by 100 makes subtle changes (like eyebrow movement) numerically distinct
            data = []
            for lm in lms:
                data.append(lm.x - nose.x)
                data.append(lm.y - nose.y)
                data.append(lm.z - nose.z)
            
            data = np.array(data) * 100 # Scaling Factor
            
            # Save file
            np.save(os.path.join(DATA_PATH, f"{count}.npy"), data)
            count += 1
            
            # Show progress
            cv2.putText(frame, f"Collecting: {count}/500", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Brief delay (Optional) to ensure frame variety
            # time.sleep(0.01) 

    cv2.imshow('ReLoop Pro Data Collector', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        collecting = True
        print("Started Collecting...")
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()