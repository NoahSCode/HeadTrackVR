import cv2
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import os

class OpticalFlowEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=5, window_size=5):
        self.threshold = threshold
        self.window_size = window_size

    def fit(self, X, y=None):
        # Not used for this estimator
        return self

    def predict(self, X):
        video_path = X[0]
        predictions = compute_optical_flow(video_path, self.threshold, self.window_size)
        return predictions


def compute_optical_flow(video_path, threshold, window_size):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file: {video_path}")

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flows_magnitudes = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, window_size, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        flows_magnitudes.append(avg_magnitude)

        prev_gray = next_gray.copy()

    cap.release()

    predictions = ['YES' if mag > threshold else 'NO' for mag in flows_magnitudes]
    return predictions


# Load the labeled data
data_path = "path_to_labels"  # Replace with your path
df = pd.read_excel(data_path)
y_true = df['Head Movement'].tolist()

# Convert YES/NO labels to 1/0
y_true_binary = [1 if label == "YES" else 0 for label in y_true]

# As we're not really doing a train-test split, set all indices to 0
test_fold = [0]*len(y_true_binary)
predefined_split = PredefinedSplit(test_fold)

video_path = "path_to_video"  # Replace with your video path

param_grid = {
    'threshold': list(range(3, 8)),
    'window_size': list(range(5, 10, 15))  # You can adjust this range based on your requirements
}

grid_search = GridSearchCV(OpticalFlowEstimator(), param_grid, cv=predefined_split, scoring='accuracy', verbose=1)
grid_search.fit([video_path] * len(y_true_binary), y_true_binary)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))