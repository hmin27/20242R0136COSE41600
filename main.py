import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import cv2
import time
import argparse
from kalman_filter import initialize_kalman_filter, kalman_filter_tracking

class PCDVisualizer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.pcd')])
        self.file_index = 0
        self.previous_pcd = None
        self.previous_centroids = None
        self.previous_tracks = {}
        self.current_view_params = None
        
        # Initialize visualizer
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.visualizer.create_window(window_name="Filtered Clusters and Bounding Boxes", width=720, height=720)
        self.view_control = self.visualizer.get_view_control()

    def process_pcd(self, file_path):
        original_pcd = o3d.io.read_point_cloud(file_path)
        voxel_size = 0.2
        downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)
        cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
        ror_pcd = downsample_pcd.select_by_index(ind)
        plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)
        final_point = ror_pcd.select_by_index(inliers, invert=True)

        labels = np.array(final_point.cluster_dbscan(eps=0.6, min_points=10, print_progress=True))
        colors = np.tile([0, 0, 1], (len(labels), 1))
        colors[labels < 0] = 0
        final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

        return final_point, labels

    def compute_centroids(self, pcd, labels):
        centroids = []
        for i in range(labels.max() + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster_points = pcd.select_by_index(cluster_indices)
            centroid = np.mean(np.asarray(cluster_points.points), axis=0)
            centroids.append((i, centroid))
        return centroids

    def update_pcd(self, vis, index):
        vis.clear_geometries()
        file_path = os.path.join(self.folder_path, self.file_names[index])
        current_pcd, labels = self.process_pcd(file_path)
        centroids_curr = self.compute_centroids(current_pcd, labels)

        if self.previous_pcd is not None and self.previous_centroids is not None:
            moving_clusters = self.detect_moving_clusters_icp(self.previous_pcd, current_pcd, self.previous_centroids, centroids_curr)
            icp_transformation = o3d.pipelines.registration.registration_icp(
                source=current_pcd,
                target=self.previous_pcd,
                max_correspondence_distance=0.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            ).transformation

            self.previous_tracks = kalman_filter_tracking(self.previous_tracks, centroids_curr, icp_transformation)

            for cluster_id in moving_clusters:
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    cluster_pcd = current_pcd.select_by_index(cluster_indices)
                    if self.is_valid_person(cluster_pcd):
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        bbox.color = (1, 0, 0)
                        vis.add_geometry(bbox)

            for track_id, kf in self.previous_tracks.items():
                position = kf.x[:3].flatten()
                cluster_indices = np.where(labels == track_id)[0]
                if len(cluster_indices) > 0:
                    cluster_pcd = current_pcd.select_by_index(cluster_indices)
                    if self.is_valid_person(cluster_pcd):                    
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        bbox.color = (0, 1, 0)
                        vis.add_geometry(bbox)
        else:
            self.previous_tracks = {}
            for cluster_id, centroid in centroids_curr:
                self.previous_tracks[cluster_id] = initialize_kalman_filter(centroid)

        vis.add_geometry(current_pcd)
        vis.get_render_option().point_size = 2.0

        if self.current_view_params is not None:
            self.view_control.convert_from_pinhole_camera_parameters(self.current_view_params, allow_arbitrary=True)

        self.previous_pcd = copy.deepcopy(current_pcd)
        self.previous_centroids = centroids_curr

    def detect_moving_clusters_icp(self, previous_pcd, current_pcd, previous_centroids, current_centroids):
        icp_threshold = 0.5
        icp_result = o3d.pipelines.registration.registration_icp(
            source=current_pcd,
            target=previous_pcd,
            max_correspondence_distance=icp_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        transformation = icp_result.transformation

        transformed_centroids = []
        for i, centroid in current_centroids:
            transformed_point = np.dot(transformation[:3, :3], centroid) + transformation[:3, 3]
            transformed_centroids.append((i, transformed_point))

        moving_clusters = []
        matched_clusters = set()

        for i_curr, transformed_centroid in transformed_centroids:
            min_distance = float('inf')
            matched_cluster = None
            for i_prev, centroid_prev in previous_centroids:
                distance = np.linalg.norm(transformed_centroid - centroid_prev)
                if distance < min_distance:
                    min_distance = distance
                    matched_cluster = i_prev
            if matched_cluster is not None and min_distance > 2.0:
                matched_clusters.add(matched_cluster)
                moving_clusters.append(i_curr)

        return moving_clusters

    def is_valid_person(self, cluster_pcd):
        points = np.asarray(cluster_pcd.points)
        num_points = len(points)

        if not (5 <= num_points <= 30):
            return False

        z_values = points[:, 2]
        z_min = z_values.min()
        z_max = z_values.max()
        height_diff = z_max - z_min

        if not (0.5 <= height_diff <= 1.0):
            return False

        # # Check the distance from the origin
        # distances = np.linalg.norm(points, axis=1)
        # if distances.max() > 30.0:
        #     return False

        return True

    def run(self):
        self.visualizer.register_key_callback(ord("N"), self.load_next_pcd)
        self.visualizer.register_key_callback(ord("P"), self.load_previous_pcd)
        self.visualizer.register_key_callback(ord("Q"), self.quit_visualizer)
        self.update_pcd(self.visualizer, 0)
        self.visualizer.run()

    def load_next_pcd(self, vis):
        self.current_view_params = self.view_control.convert_to_pinhole_camera_parameters()
        if self.file_index < len(self.file_names) - 1:
            self.file_index += 1
            self.update_pcd(vis, self.file_index)
        return False

    def load_previous_pcd(self, vis):
        self.current_view_params = self.view_control.convert_to_pinhole_camera_parameters()
        if self.file_index > 0:
            self.file_index -= 1
            self.update_pcd(vis, self.file_index)
        return False

    def quit_visualizer(self, vis):
        vis.close()
        return False

    def save_image(self, folder_path):

        folder_name = os.path.basename(os.path.dirname(folder_path))
        output_image_path = f"image/{folder_name}"

        self.visualizer.update_renderer()

        start_time = time.time()

        # 각 PCD 파일 시각화 및 비디오 저장
        while self.file_index < len(self.file_names):
            current_view_params = self.view_control.convert_to_pinhole_camera_parameters()

            self.update_pcd(self.visualizer, self.file_index)
            self.visualizer.poll_events()
            self.visualizer.update_renderer()

            img = np.asarray(self.visualizer.capture_screen_float_buffer(do_render=True))
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            image_filename = f"{output_image_path}_{self.file_index + 1:04d}.png"
            cv2.imwrite(image_filename, img)
            
            elapsed_time = time.time() - start_time
            print(f"Processed {self.file_index + 1}/{len(self.file_names)} files. Elapsed time: {elapsed_time:.2f} seconds")

            self.file_index += 1

        self.visualizer.destroy_window()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PCD files.")
    parser.add_argument('folder_path', type=str, help='Path to the folder containing PCD files')
    args = parser.parse_args()

    render = PCDVisualizer(args.folder_path)
    # render.save_image(args.folder_path)
    render.run()
