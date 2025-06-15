# extract_all_skeleton_to_npy.py

import cv2
import mediapipe as mp
import numpy as np
import os

# 설정
base_video_folder = '/home/yunddu/Semicon_Project/Dataset'        # 원본 비디오 경로
base_output_folder = '/home/yunddu/Semicon_Project/npy_data'      # npy 저장 경로
sequence_length = 30

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# joint 개수 (BlazePose 기준)
N_joint = 33

# 비디오 처리 함수
def process_video(video_path, output_path, sequence_length):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    skeleton_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append(lm.x)
                keypoints.append(lm.y)
            skeleton_sequence.append(keypoints)
            frame_count += 1

        if frame_count >= sequence_length:
            break

    cap.release()

    # 🚨 skeleton_sequence가 비어있는 경우 → skip 처리
    if len(skeleton_sequence) == 0:
        print(f"[⚠️] Skipping: {video_path} (no skeleton detected!)")
        return  # 저장하지 않고 skip

    # 시퀀스 길이 맞추기
    if len(skeleton_sequence) < sequence_length:
        print(f"[⚠️] Padding: {video_path}, frames={len(skeleton_sequence)}")
        while len(skeleton_sequence) < sequence_length:
            skeleton_sequence.append(skeleton_sequence[-1])
    else:
        skeleton_sequence = skeleton_sequence[:sequence_length]

    skeleton_array = np.array(skeleton_sequence)
    np.save(output_path, skeleton_array)
    print(f"[💾] Saved: {output_path} | shape={skeleton_array.shape}")

# 전체 구조 한 번에 돌리기
splits = ['train', 'val']
classes = ['daily', 'falldown']

for split in splits:
    for cls in classes:
        input_video_folder = os.path.join(base_video_folder, split, cls)
        output_npy_folder = os.path.join(base_output_folder, split, cls)

        os.makedirs(output_npy_folder, exist_ok=True)

        print(f"\n[▶️] Processing {split}/{cls} ...")
        file_list = [f for f in os.listdir(input_video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        if len(file_list) == 0:
            print(f"[⚠️] No video files found in {input_video_folder}")
            continue

        for file in file_list:
            video_path = os.path.join(input_video_folder, file)
            video_name = os.path.splitext(file)[0]
            output_path = os.path.join(output_npy_folder, f'{video_name}.npy')
            process_video(video_path, output_path, sequence_length)

print("\n[✅] All videos processed for train/val daily/falldown!")
