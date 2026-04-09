# coding=utf-8
import os
import cv2


def video2jpg(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True
    print('read video in', video_path)
    videoclass=os.path.basename(video_path).split('.')[0]
    save_jpg_path=os.path.join(os.path.dirname(video_path),videoclass)
    if not os.path.exists(save_jpg_path):
        os.makedirs(save_jpg_path)
        print('make dir {}'.format(save_jpg_path))

    while success:
        success, frame = cap.read()
        if not success:
            break
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        #print("Frame %d" %frame_count)
        cv2.imwrite(save_jpg_path + "/{}I{}.jpg".format(videoclass,str(frame_count).zfill(4)), frame, params)
        frame_count = frame_count + 1
    cap.release()


def get_all_dir(root_dir):
    video_pathes = []
    for dir in os.listdir(root_dir):
        dir = os.path.join(root_dir, dir)
        if os.path.isdir(dir):
            video_pathes.extend(get_all_dir(dir))
        if dir[-4:] == '.mp4':
            video_pathes.append(dir)
    return video_pathes


root_dir = './datasets/Anti-UAV-RGBT'
video_pathes = get_all_dir(root_dir)

for vid, video_path in enumerate(video_pathes):
    video2jpg(video_path)