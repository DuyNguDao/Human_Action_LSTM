import cv2
from yolov7_pose.detect_pose import Y7Detect, draw_kpts, draw_boxes
import time
import numpy as np
import math
from numpy import random
from track_sort.Sort import SORT
from strong_sort.strong_sort import StrongSORT
from pathlib import Path
from collections import deque
import torch
import argparse
from classification_lstm.utils.load_model import Model
from classification_stgcn.Actionsrecognition.ActionsEstLoader import TSSTG
import random

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(torch.cuda.is_available())


def detect_video(url_video=None, name_model=None, flag_save=False, fps=None, name_video='video.avi'):

    # ******************************** LOAD MODEL *************************************************
    # load model detect yolov7
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    y7_model = Y7Detect()
    class_name = y7_model.class_names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_name]
    # *************************** LOAD MODEL LSTM OR ST-GCN ************************************************
    if name_model=='lstm':
        # LSTM
        action_model = Model(device=device, skip=True)
    else:
        # ST-GCN
        action_model = TSSTG(device=device, skip=True)
    # **************************** INIT TRACKING *************************************************
    tracker = StrongSORT(device=device, max_age=30, n_init=3, max_iou_distance=0.7)  # deep sort
    # tracker = SORT()
    # ********************************** READ VIDEO **********************************************
    if url_video == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(url_video)

    # get size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    h_norm, w_norm = 720, 1280
    if frame_height > h_norm or frame_width > w_norm:
        frame_width = w_norm
        frame_height = h_norm
    # get fps of camera
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    # save video
    if flag_save is True:
        video_writer = cv2.VideoWriter(name_video,
                                       cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # ******************************** REAL TIME ********************************************
    memory = {}
    count = True
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape

        if h > h_norm or w > w_norm:
            rate_max = max(h_norm / h, w_norm / w)
            frame = cv2.resize(frame, (int(rate_max * w), int(rate_max * h)), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape
        # ************************ DETECT YOLOv5 ***************************************
        if count:
            bbox, labels, score, lables_id, kpts = y7_model.predict(frame)
            bbox, score, kpts = np.array(bbox), np.array(score), np.array(kpts)

        if len(bbox) != 0:
            if count:
                data = tracker.update(bbox, score, kpts, frame)
            for outputs in data:
                if len(outputs['bbox']) != 0:
                    box, kpt, track_id, list_kpt = outputs['bbox'], outputs['kpt'], outputs['id'],\
                                                             outputs['list_kpt']
                    # if track_id not in memory:
                    #     memory[track_id] = deque(maxlen=30)
                    # memory[track_id].append(kpt)
                    # if len(memory[track_id]) == 30:
                    #     list_kpt = np.array(memory[id], dtype=np.float32)
                    #     action, score = action_model.predict([list_kpt], w, h, batch_size=5)
                    # draw_boxes(frame, box, color=colors[icolor])
                    kpt = kpt[:, :2].astype('int')
                    draw_kpts(frame, [kpt])
                    color = (0, 255, 0)
                    if len(list_kpt) == 15:
                        if name_model=='lstm':
                            # LSTM
                            action, score = action_model.predict([list_kpt], w, h, batch_size=1)
                            score = score[0]
                        else:
                            # ST-GCN
                            action, score = action_model.predict(list_kpt, image_size=[w, h])
                    try:
                        if action[0] == "Fall Down":
                            color = (0, 0, 255)
                        cv2.putText(frame, '{}: {}% - {}'.format(action[0], round(score), track_id),
                                    (max(box[0]-20, 0), box[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                        action = ["Pending..."]
                    except:
                        cv2.putText(frame, '{}: {}% - {}'.format("Pending...", round(0), track_id),
                                    (max(box[0]-20, 0), box[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        count = not count
        # ******************************************** SHOW *******************************************
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        fps = int(1 / (time.time() - start))
        cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        if flag_save is True:
            video_writer.write(frame)

    cap.release()
    if flag_save is True:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face On Video')
    parser.add_argument("-fn", "--file_name", help="video file name or rtsp", default='', type=str)
    parser.add_argument("-nm", "--name_model", help="lstm or stgcn", default='stgcn', type=str)
    parser.add_argument("-op", "--option", help="if save video then choice option = True", default=False, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='recog_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=30, help="FPS of output video", type=int)
    args = parser.parse_args()

    # PATH VIDEO
    # url = '/home/duyngu/Downloads/video_test/20221001153808324_7F01683RAZE9C1D.mp4'
    source = args.file_name
    cv2.namedWindow('video')
    # if run  as terminal, replace url = source
    detect_video(url_video=source, name_model=args.name_model,
              flag_save=args.option, fps=args.fps, name_video=args.output)