import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import argparse
from filterpy.kalman import KalmanFilter
import numpy as np
import time

parser = argparse.ArgumentParser(description="Process PCD files.")
parser.add_argument('folder_path', type=str, help='Path to the folder containing PCD files')
args = parser.parse_args()

# PCD 데이터 로드
folder_path = args.folder_path
file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.pcd')])
file_index = 0

# VisualizerWithKeyCallback 생성 및 창 열기
visualizer = o3d.visualization.VisualizerWithKeyCallback()
visualizer.create_window(window_name="Filtered Clusters and Bounding Boxes", width=720, height=720)

# 카메라 뷰포인트 저장 변수
view_control = visualizer.get_view_control()
current_view_params = None

# 이전 PCD 저장 변수
previous_pcd = None
previous_centroids = None
previous_tracks = {}

# Kalman Filter 초기화 함수
def initialize_kalman_filter(centroid):
    """
    Initialize a Kalman Filter for a given cluster centroid.
    """
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

    # 공분산 행렬
    kf.R = np.eye(3) * 0.1
    kf.Q = np.eye(6) * 0.1 
    kf.P *= 1000.0

    # 초기 상태값
    kf.x[:3] = centroid.reshape(-1, 1)
    kf.x[3:] = 0.0 

    return kf

# Kalman Filter 기반 객체 추적 함수
def kalman_filter_tracking(previous_tracks, current_centroids, icp_transformation):

    updated_tracks = {}

    for track_id, kf in previous_tracks.items():
        kf.predict()
        
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

        if best_track_id is not None and min_distance < 2.0:  # track 업데이트
            previous_tracks[best_track_id].update(centroid)
            updated_tracks[best_track_id] = previous_tracks[best_track_id]
        else:
            # 새로운 item 등록
            new_track_id = len(previous_tracks) + len(updated_tracks) + 1
            updated_tracks[new_track_id] = initialize_kalman_filter(centroid)

    return updated_tracks


# 클러스터링 및 필터링을 수행하는 함수
def process_pcd(file_path):
    original_pcd = o3d.io.read_point_cloud(file_path)

    # Voxel Downsampling 수행
    voxel_size = 0.2
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Radius Outlier Removal (ROR) 적용
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    # RANSAC을 사용하여 평면 제거
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                                 ransac_n=3,
                                                 num_iterations=2000)
    final_point = ror_pcd.select_by_index(inliers, invert=True)

    # DBSCAN 클러스터링 적용
    labels = np.array(final_point.cluster_dbscan(eps=0.6, min_points=10, print_progress=True))
    colors = np.tile([0, 0, 1], (len(labels), 1))  # 파란색
    colors[labels < 0] = 0  # 노이즈는 검정색
    final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return final_point, labels

# 이동한 객체를 ICP로 클러스터 단위로 검출하는 함수
def detect_moving_clusters_icp(previous_pcd, current_pcd, previous_centroids, current_centroids):
    # ICP registration
    icp_threshold = 0.5
    icp_result = o3d.pipelines.registration.registration_icp(
        source=current_pcd,
        target=previous_pcd,
        max_correspondence_distance=icp_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation = icp_result.transformation

    # 현재 centroids 변환
    transformed_centroids = []
    for i, centroid in current_centroids:
        transformed_point = np.dot(transformation[:3, :3], centroid) + transformation[:3, 3]
        transformed_centroids.append((i, transformed_point))

    moving_clusters = []
    matched_clusters = set()

    # Matching
    for i_curr, transformed_centroid in transformed_centroids:
        min_distance = float('inf')
        matched_cluster = None
        for i_prev, centroid_prev in previous_centroids:
            distance = np.linalg.norm(transformed_centroid - centroid_prev)
            if distance < min_distance:
                min_distance = distance
                matched_cluster = i_prev
        if matched_cluster is not None and min_distance > 2.0:  # 이동 거리 임계값
            matched_clusters.add(matched_cluster)
            moving_clusters.append(i_curr)

    return moving_clusters

# 각 클러스터의 중심 좌표 계산 함수
def compute_centroids(pcd, labels):
    centroids = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_points = pcd.select_by_index(cluster_indices)
        centroid = np.mean(np.asarray(cluster_points.points), axis=0)
        centroids.append((i, centroid))
    return centroids

# 사람에 대한 필터링 조건 명시
def is_valid_person(cluster_pcd):
    bbox = cluster_pcd.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()  # (x, y, z)
    height, width, depth = bbox_extent[2], bbox_extent[0], bbox_extent[1]

    if 1.5 <= height <= 2.0 and 0.2 <= width <= 1.0 and 0.2 <= depth <= 1.0:
        return True
    return False

# Callback Function 수정
def update_pcd(vis, index, view_control, viewpoint_params):
    global previous_pcd, previous_centroids, previous_tracks

    vis.clear_geometries()
    file_path = os.path.join(folder_path, file_names[index])
    current_pcd, labels = process_pcd(file_path)

    centroids_curr = compute_centroids(current_pcd, labels)

    if previous_pcd is not None and previous_centroids is not None:
        # ICP를 통해 이동 클러스터 검출
        moving_clusters = detect_moving_clusters_icp(previous_pcd, current_pcd, previous_centroids, centroids_curr)

        icp_transformation = o3d.pipelines.registration.registration_icp(
            source=current_pcd,
            target=previous_pcd,
            max_correspondence_distance=0.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        ).transformation

        # Kalman Filter로 트래킹 업데이트
        previous_tracks = kalman_filter_tracking(previous_tracks, centroids_curr, icp_transformation)

        # 이동 클러스터에 대해 바운딩 박스 생성
        for cluster_id in moving_clusters:
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > 0:  # 포인트가 존재하는 경우
                cluster_pcd = current_pcd.select_by_index(cluster_indices)
                if is_valid_person(cluster_pcd):  # 유효한 사람인지 확인
                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                    bbox.color = (1, 0, 0)  # 이동 클러스터는 빨간색
                    vis.add_geometry(bbox)

        # Kalman Filter 추적된 객체에 대해 바운딩 박스 생성
        for track_id, kf in previous_tracks.items():
            position = kf.x[:3].flatten()
            cluster_indices = np.where(labels == track_id)[0]
            if len(cluster_indices) > 0:  # 포인트가 존재하는 경우
                cluster_pcd = current_pcd.select_by_index(cluster_indices)
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                bbox.color = (0, 1, 0)  # 추적된 객체는 초록색
                vis.add_geometry(bbox)
    else:
        # 첫 번째 프레임: Kalman Filter 초기화
        previous_tracks = {}
        for cluster_id, centroid in centroids_curr:
            previous_tracks[cluster_id] = initialize_kalman_filter(centroid)

    vis.add_geometry(current_pcd)
    vis.get_render_option().point_size = 2.0

    if viewpoint_params is not None:
        view_control.convert_from_pinhole_camera_parameters(viewpoint_params, allow_arbitrary=True)

    previous_pcd = copy.deepcopy(current_pcd)
    previous_centroids = centroids_curr

    del current_pcd



def load_next_pcd(vis):
    global file_index, current_view_params

    current_view_params = view_control.convert_to_pinhole_camera_parameters()

    if file_index < len(file_names) - 1:
        file_index += 1
        update_pcd(vis, file_index, view_control, current_view_params)

    return False

def load_previous_pcd(vis):
    global file_index, current_view_params

    current_view_params = view_control.convert_to_pinhole_camera_parameters()

    if file_index > 0:
        file_index -= 1
        update_pcd(vis, file_index, view_control, current_view_params)

    return False

def quit_visualizer(vis):
    vis.close()
    return False

def auto_advance_pcd():
    global file_index, current_view_params
    while visualizer.poll_events():
        current_view_params = view_control.convert_to_pinhole_camera_parameters()
        if file_index < len(file_names) - 1:
            file_index += 1
        else:
            file_index = 0
        update_pcd(visualizer, file_index, view_control, current_view_params)
        visualizer.update_renderer()
        time.sleep(0.7)
        
visualizer.register_key_callback(ord("N"), load_next_pcd)
visualizer.register_key_callback(ord("P"), load_previous_pcd)
visualizer.register_key_callback(ord("Q"), quit_visualizer)

# 첫 번째 파일 로드
update_pcd(visualizer, file_index, view_control, current_view_params)
current_view_params = view_control.convert_to_pinhole_camera_parameters()

# auto_advance_pcd()
visualizer.run()
visualizer.destroy_window()
