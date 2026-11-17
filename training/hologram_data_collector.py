import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ==========================================
# Recording Settings - Change the name per word
# ==========================================
WORD_NAME = "Life"  # The word you want to record (e.g., "Hello", "Yes", "No", etc.)
RECORD_DURATION_SECONDES = 10  # Recording duration of 5 full natural seconds
FPS = 20  # Desired frame rate

if not os.path.exists('videos'):
    os.makedirs('videos')

video_filename = f"videos/{WORD_NAME.lower().replace(' ', '_')}.mp4"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Using 'mp4v' or 'avc1' codec - Settings adjusted to ensure consistent timing
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, FPS, (width, height))

print("==================================================")
print(f"🎬 SYSTEM: Preparing to record Natural Avatar for: '{WORD_NAME.upper()}'")
print("==================================================")

# Standard countdown loop
for i in range(5, 0, -1):
    ret, frame = cap.read()
    if not ret: break
    cv2.putText(frame, f"GET READY... RECORDING IN {i}", (50, height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Avatar Recorder Studio", frame)
    cv2.waitKey(1000)

print("🔴 RECORDING STARTED! Move naturally...")

total_frames = RECORD_DURATION_SECONDES * FPS
recorded_frames = 0

# Target duration per frame in seconds (e.g., 1000ms / 20fps = 50ms per frame)
time_per_frame = 1.0 / FPS 

while recorded_frames < total_frames:
    start_time = time.time() # Track the start time of the current frame processing
    
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    avatar_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                avatar_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 200), thickness=3, circle_radius=4), # Increased thickness slightly for visibility
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )
            
    cv2.putText(avatar_frame, f"RELOOP ACADEMY: {WORD_NAME.upper()}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)
                
    out.write(avatar_frame)
    recorded_frames += 1
    
    cv2.imshow("Avatar Recorder Studio", avatar_frame)
    
    # 🧠 Smart speed synchronization to match real-time pace:
    process_time = time.time() - start_time # Actual processing duration for this frame
    delay_needed = time_per_frame - process_time # Remaining time to complete the target frame step duration
    
    if delay_needed > 0:
        time.sleep(delay_needed) # Enforce delay to perfectly align with natural real-time speed
        
    if cv2.waitKey(1) & 0xFF == 27: # Press ESC key to exit
        print("❌ Recording cancelled.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("==================================================")
print(f"✅ SUCCESS: Natural Speed Avatar Saved!")
print(f"📁 Path: {video_filename}")
print("==================================================")