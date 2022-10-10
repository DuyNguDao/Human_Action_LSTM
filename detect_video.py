import cv2
from yolov7_pose.detect_pose import Y7Detect, draw_kpts, draw_boxes
import time
import numpy as np
import math
from numpy import random
from TRACK_SORT.Sort import SORT
from strong_sort.strong_sort import StrongSORT
from pathlib import Path
from collections import deque
import torch
import argparse
from classifcation_lstm.utils.load_model import Model

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_cached()
print(torch.cuda.is_available())


def detect_video(url_video=None, path_model=None, flag_save=False, fps=None, name_video='video.avi'):

    # ******************************** LOAD MODEL *************************************************
    # load model detect yolov5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    y7_model = Y7Detect(weights=path_model)
    class_name = y7_model.class_names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_name]

    # *************************** LOAD MODEL LSTM ************************************************
    action_model = Model('runs/exp4/best.pt')
    # **************************** INIT TRACKING *************************************************
    tracker = StrongSORT(device=device)
    # ********************************** READ VIDEO **********************************************
    if url_video == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(url_video)

    # get size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    h_norm, w_norm = 720, 1280
    if frame_height > h_norm and frame_width > w_norm:
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
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        if h > h_norm and w > w_norm:
            frame = cv2.resize(frame, (w_norm, h_norm), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape
        # frame[0:h-400, w-300:w] = np.zeros((h-400, 300, 3), dtype='uint8')
        # frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
        # h, w, _ = frame.shape
        # ************************ DETECT YOLOv5 ***************************************
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox, label, score, label_id, kpts, scores_pt, line_pt = y7_model.predict(image)
        bbox, score, kpts = np.array(bbox), np.array(score), np.array(kpts)
        if len(bbox) != 0:
            data = tracker.update(bbox, score, kpts, line_pt, frame)
            for outputs in data:
                if len(outputs['bbox']) != 0:
                    box, kpt, line_kpt, track_id, list_kpt = outputs['bbox'], outputs['kpt'],\
                                                             outputs['line_kpt'], outputs['id'],\
                                                             outputs['list_kpt']
                    # if track_id not in memory:
                    #     memory[track_id] = deque(maxlen=30)
                    # memory[track_id].append(kpt)
                    # if len(memory[track_id]) == 30:
                    #     list_kpt = np.array(memory[id], dtype=np.float32)
                    #     action, score = action_model.predict([list_kpt], w, h, batch_size=5)
                    icolor = class_name.index('person')
                    draw_boxes(frame, box, color=colors[icolor])
                    draw_kpts(frame, [kpt], [line_kpt])
                    color = (255, 0, 0)
                    if len(list_kpt) == 30:
                        action, score = action_model.predict([list_kpt], w, h, batch_size=5)
                    try:
                        if action[0] == "Fall Down":
                            color = (0, 0, 255)
                        cv2.putText(frame, '{}: {}% - {}'.format(action[0], round(score[0]), track_id), (box[0], box[1]+ 20),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                        action = ["Pending..."]
                    except:
                        cv2.putText(frame, '{}: {}% - {}'.format("Pending...", round(0), track_id), (box[0], box[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

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
    parser.add_argument("-op", "--option", help="if save video then choice option = 1", default=False, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='recog_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=20, help="FPS of output video", type=int)
    args = parser.parse_args()

    # MODEL YOLOV5
    path_models = '/home/duyngu/Desktop/LSTM_Interface/yolov7_pose/weights/yolov7-w6-pose.pt'
    # PATH VIDEO
    url = '/home/duyngu/Downloads/video_test/20221001152536279_7F01683RAZE9C1D.mp4'
    source = args.file_name
    cv2.namedWindow('video')
    # if run  as terminal, replace url = source
    detect_video(url_video=url, path_model=path_models,
              flag_save=args.option, fps=args.fps, name_video=args.output)