import cv2
from cv2 import optflow
import os
import pandas as pd
import numpy as np
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_counter = 0
    saved_frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % (fps // 5) == 0:
            frame_filename = os.path.join(output_folder, f"frame{saved_frame_counter}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_counter += 1

        frame_counter += 1

    cap.release()



def generate_timestamps(output_folder):
    frames = os.listdir(output_folder)
    num_sets = len(frames) // 5  # Number of sets of 5 frames
    
    timestamps = [i for i in range(num_sets)]
    
    # Convert timestamps to hh:mm:ss format
    formatted_timestamps = [f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}" for t in timestamps]

    return formatted_timestamps
def compute_optical_flow(output_folder, timestamps):
    flows_magnitudes = []
    frames = sorted(os.listdir(output_folder))

    for i in range(0, len(frames) - 4, 5):  # Step by 5 frames
        flow_values_for_set = []

        for j in range(4):  # For the 4 optical flow calculations in the set
            frame1_path = os.path.join(output_folder, frames[i+j])
            frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)

            frame2_path = os.path.join(output_folder, frames[i+j+1])
            frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = np.mean(magnitude)

            flow_values_for_set.append(avg_magnitude)
        
        flows_magnitudes.append(np.mean(flow_values_for_set))  # Average for the set

    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Optical Flow Magnitude': flows_magnitudes
    })

    df.to_excel(os.path.join(output_folder, "optical_flow_data.xlsx"), index=False)

def compute_and_display_optical_flow(video_path, output_folder, threshold=5.0, window_size=5):
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flows_magnitudes = []
    head_movements = []  # List to store 'YES'/'NO' values for head movement
    frame_timestamps = []

    frame_number = 0  # Track frame number for timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    
    window = []  # List to store recent optical flow magnitudes for temporal filtering

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        
        window.append(avg_magnitude)
        if len(window) > window_size:
            window.pop(0)  # Remove oldest magnitude if window size exceeded
        smoothed_flow = np.mean(window)  # Take the average of the magnitudes in the window
        
        flows_magnitudes.append(smoothed_flow)
        
        frame_timestamps.append(frame_number / fps)  # Store the timestamp
        frame_number += 1
        
        overlay = next_frame.copy()
        cv2.putText(overlay, f"Flow: {smoothed_flow:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Check threshold and display the message accordingly
        if smoothed_flow > threshold:
            movement_message = "Head movement detected: YES"
            head_movements.append("YES")
        else:
            movement_message = "Head movement detected: NO"
            head_movements.append("NO")
        
        cv2.putText(overlay, movement_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # green color

        cv2.imshow('Optical Flow', overlay)
        if cv2.waitKey(10) & 0xFF == 27:
            break

        prev_gray = next_gray.copy()

    cv2.destroyAllWindows()

    # Convert frame timestamps into hh:mm:ss format
    timestamps = ["{:02d}:{:02d}:{:02d}.{:03d}".format(int(t // 3600), int((t % 3600) // 60), int(t % 60), int((t % 1) * 1000)) for t in frame_timestamps]
    
    # Save the data to an Excel file
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Optical Flow': flows_magnitudes,
        'Head Movement': head_movements  # new column
    })

    df.to_excel(os.path.join(output_folder, 'optical_flow_data.xlsx'), index=False)

    return

if __name__ == "__main__":
    video_path = 'path_to_video'
    output_folder = 'path_to_output_folder'
    timestamps = generate_timestamps(output_folder)
    compute_and_display_optical_flow(video_path, output_folder, threshold=6, window_size=5)