import cv2
import os
import pandas as pd

def create_annotation_template(video_path, output_folder):
    # Create a video capture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return
    
    frame_timestamps = []
    frame_number = 0  # Track frame number for timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    
    # Iterate over the frames just to calculate timestamps
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_timestamps.append(frame_number / fps)  # Store the timestamp
        frame_number += 1

    cap.release()

    # Convert frame timestamps into hh:mm:ss format
    timestamps = [
    "{:02d}:{:02d}:{:02d}.{:03d}".format(
        int(t // 3600),
        int((t % 3600) // 60),
        int(t % 60),
        int((t % 1) * 1000)
    )
    for t in frame_timestamps
]
    
    # Create a DataFrame with timestamps and empty "head movement" column
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Head Movement': [''] * len(timestamps)
    })

    # Save the DataFrame to an Excel file
    df.to_excel(os.path.join(output_folder, 'annotation_template_test.xlsx'), index=False)

video_path = 'path_to_video_file'
output_folder = 'path_to_output_folder'
create_annotation_template(video_path, output_folder)