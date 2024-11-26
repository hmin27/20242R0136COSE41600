import numpy as np
from filterpy.kalman import KalmanFilter

def initialize_kalman_filter(centroid):
    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = 1.0  # Time step

    kf.F = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    # Measurement noise covariance
    kf.R = np.eye(3) * 0.1
    # Process noise covariance
    kf.Q = np.eye(6) * 0.1 
    # Initial state covariance
    kf.P *= 1000.0

    # Initial state
    kf.x[:3] = centroid.reshape(-1, 1)
    kf.x[3:] = 0.0 

    return kf

# Kalman Filter tracking update function
def kalman_filter_tracking(previous_tracks, current_centroids, icp_transformation):
    """
    Update the tracking of centroids using Kalman Filters.
    """
    updated_tracks = {}

    # Predict step for all existing tracks
    for track_id, kf in previous_tracks.items():
        kf.predict()

    # Transform centroids using the ICP transformation
    transformed_centroids = []
    for cluster_id, centroid in current_centroids:
        transformed_point = np.dot(icp_transformation[:3, :3], centroid) + icp_transformation[:3, 3]
        transformed_centroids.append((cluster_id, transformed_point))

    # Update step
    for cluster_id, centroid in current_centroids:
        min_distance = float('inf')
        best_track_id = None
        for track_id, kf in previous_tracks.items():
            predicted_position = kf.x[:3].flatten()
            distance = np.linalg.norm(predicted_position - centroid)
            if distance < min_distance:
                min_distance = distance
                best_track_id = track_id

        if best_track_id is not None and min_distance < 2.0:  # Update existing track
            previous_tracks[best_track_id].update(centroid)
            updated_tracks[best_track_id] = previous_tracks[best_track_id]
        else:  # Create new track
            new_track_id = len(previous_tracks) + len(updated_tracks) + 1
            updated_tracks[new_track_id] = initialize_kalman_filter(centroid)

    return updated_tracks
