import cv2
import os
import argparse


def images_to_video(image_folder, output_video_path, frame_rate=10):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    
    if not images:
        raise ValueError("No images found in the folder.")

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PCD files.")
    parser.add_argument('file_name', type=str, help='The name of scenario')
    args = parser.parse_args()

    image_folder = f"image/{args.file_name}"  # Path to the folder containing images
    output_video_path = f"result/{args.file_name}.mp4"  # Path where the output video will be saved
    frame_rate = 10  # Frame rate of the output video

    images_to_video(image_folder, output_video_path, frame_rate)
