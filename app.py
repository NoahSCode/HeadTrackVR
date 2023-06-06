import cv2
import numpy as np
import pandas as pd
from datetime import datetime


# Create a VideoCapture object
cap = cv2.VideoCapture('Recordings/drone.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Get frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get total number of frames in the video
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Calculate total video duration
total_duration = (total_frames * (1/fps)) * 1000  # in milliseconds

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

# Convert the first frame to grayscale
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Define the movement threshold
threshold = 10000000  # you may need to adjust this value

# Initialize an empty DataFrame to store the movement log
movement_log = pd.DataFrame(columns=['Time', 'Movement Detected'])

# Initialize total movement duration
total_movement_duration = 0

# Read until video is completed
while cap.isOpened():
    # Capture the next frame
    ret, curr_frame = cap.read()

    # Break the loop if the frame couldn't be captured
    if not ret:
        break

    # Convert the current frame to grayscale
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current and previous frame
    diff_frame = cv2.absdiff(prev_frame, curr_frame)

    # Compute the sum of differences
    movement = np.sum(diff_frame)

    # Get current frame number
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # Calculate video timestamp in milliseconds
    timestamp = (frame_number * (1/fps)) * 1000

    # Append the timestamp and movement value to the log
    movement_log = movement_log.append({'Time': timestamp, 'Value': movement}, ignore_index=True)

    # If the movement exceeds the threshold, mark that movement is detected
    if movement > threshold:
        # Increment total movement duration
        total_movement_duration += (1/fps) * 1000

        # Mark 'YES' for movement detected
        movement_log.loc[movement_log['Time'] == timestamp, 'Movement Detected'] = 'YES'
    else:
        # Mark 'NO' for movement detected
        movement_log.loc[movement_log['Time'] == timestamp, 'Movement Detected'] = 'NO'

    # Display the difference frame
    cv2.imshow('Difference Frame', diff_frame)

    # Make the current frame the previous frame for the next iteration
    prev_frame = curr_frame.copy()

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# After reading the video file, release the capture object and close all frames
cap.release()
cv2.destroyAllWindows()

# Function to convert time in milliseconds to mm:ss:ms format
def format_time(ms):
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return "{:02}:{:02}:{:03}".format(int(minutes), int(seconds), int(ms))

# Create a filtered DataFrame where 'Movement Detected' is 'YES'
filtered_movement_log = movement_log

# Apply 'format_time' function to 'Time' column of filtered_movement_log
filtered_movement_log['Time'] = filtered_movement_log['Time'].apply(format_time)

# Update the original movement_log with the formatted 'Time' from filtered_movement_log
movement_log.update(filtered_movement_log)

# Append total time and total movement time to the log
summary = pd.DataFrame({
    'Total Video Time': [format_time(total_duration)],
    'Total Movement Time': [format_time(total_movement_duration)]
})

# Save the movement log and summary to an Excel file
with pd.ExcelWriter('movement_log.xlsx') as writer:
    movement_log.to_excel(writer, sheet_name='Movement Log', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)