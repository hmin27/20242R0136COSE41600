import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

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

    kf.R = np.eye(3) * 0.1
    kf.Q = np.eye(6) * 0.1 
    kf.P *= 1000.0

    # Initial state
    kf.x[:3] = centroid.reshape(-1, 1)
    kf.x[3:] = 0.0 

    return kf

def kalman_filter_tracking(previous_tracks, current_centroids, icp_transformation):
    updated_tracks = {}

    for track_id, kf in previous_tracks.items():
        kf.predict()

    # Transform centroids using the ICP transformation
    transformed_centroids = []
    for cluster_id, centroid in current_centroids:
        transformed_point = np.dot(icp_transformation[:3, :3], centroid) + icp_transformation[:3, 3]
        transformed_centroids.append((cluster_id, transformed_point))

    # Create a cost matrix
    cost_matrix = []
    track_ids = list(previous_tracks.keys())
    for cluster_id, centroid in current_centroids:
        cost_row = []
        for track_id in track_ids:
            predicted_position = previous_tracks[track_id].x[:3].flatten()
            distance = np.linalg.norm(predicted_position - centroid)
            cost_row.append(distance)
        cost_matrix.append(cost_row)

    cost_matrix = np.array(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assigned_tracks = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 2.0: 
            cluster_id = current_centroids[r][0]
            track_id = track_ids[c]
            centroid = current_centroids[r][1]
            previous_tracks[track_id].update(centroid)
            previous_tracks[track_id].frames_unassigned = 0
            updated_tracks[track_id] = previous_tracks[track_id]
            assigned_tracks.add(track_id)

    # Handle unassigned tracks
    for track_id in track_ids:
        if track_id not in assigned_tracks:
            if not hasattr(previous_tracks[track_id], 'frames_unassigned'):
                previous_tracks[track_id].frames_unassigned = 1
            else:
                previous_tracks[track_id].frames_unassigned += 1
            
            if previous_tracks[track_id].frames_unassigned <= 5:
                updated_tracks[track_id] = previous_tracks[track_id]

    # Create new tracks for unassigned centroids
    for r in range(len(current_centroids)):
        if r not in row_ind:
            cluster_id, centroid = current_centroids[r]
            new_track_id = len(previous_tracks) + len(updated_tracks) + 1
            new_track = initialize_kalman_filter(centroid)
            new_track.frames_unassigned = 0
            updated_tracks[new_track_id] = new_track

    # Velocity  moving average
    moving_tracks = {}
    velocity_threshold = 0.001
    for track_id, kf in updated_tracks.items():
        velocity = np.linalg.norm(kf.x[3:].flatten())

        if hasattr(kf, 'velocity_history'):
            kf.velocity_history.append(velocity)
            if len(kf.velocity_history) > 3:
                kf.velocity_history.pop(0)
            avg_velocity = np.mean(kf.velocity_history)
        else:
            kf.velocity_history = [velocity]
            avg_velocity = velocity

        # Only keep tracks with significant movement
        if avg_velocity > velocity_threshold:
            moving_tracks[track_id] = kf

    return moving_tracks
