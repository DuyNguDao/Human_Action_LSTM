import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2
from classifcation_lstm.models.rnn import RNN
import csv
import numpy as np


class Model:
    def __init__(self, path):
        # config device cuda or cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RNN(input_size=26, num_classes=7, device=self.device).to(self.device)
        # self.model = CNNLSTM(num_classes=7).to(self.device)
        self.path = path
        self.load_model()
        self.model.eval()
        self.class_names = ['Standing', 'Stand up', 'Sitting','Sit down','Lying Down','Walking','Fall Down']

    def load_model(self):
        """
        function: load model and parameter
        :return:
        """
        # load model
        self.model.load_state_dict(torch.load(self.path, map_location=self.device))

    def preprocess_data(self, list_data, size_w, size_h):
        """
        function: preprocessing image
        :param image: array image
        :return:
        """

        def scale_pose(xy):
            """
            Normalize pose points by scale with max/min value of each pose.
            xy : (frames, parts, xy) or (parts, xy)
            """
            if xy.ndim == 2:
                xy = np.expand_dims(xy, 0)
            xy_min = np.nanmin(xy, axis=1)
            xy_max = np.nanmax(xy, axis=1)
            for i in range(xy.shape[0]):
                xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
            return xy.squeeze()

        pose = np.array(list_data)
        pose = np.concatenate([pose[:, :, 0:1, :], pose[:, :, 5:, :]], axis=2) # remove point 1,2,3,4
        # normalize
        pose[:, :, :, 0] /= size_w
        pose[:, :, :, 1] /= size_h
        list_template = []
        for action in pose:
            action = scale_pose(action)
            action = action.reshape(len(action), 26)
            action = torch.tensor(action)
            list_template.append(action)
        return list_template

    def predict(self, list_data, size_w, size_h, batch_size=5):
        """
        function: predict image
        :param image: array image bgr
        :return: name class predict and list prob predict
        """
        import math
        list_data = self.preprocess_data(list_data, size_w, size_h)
        label, score = [], []
        for i in range(math.ceil(len(list_data)/batch_size)):
            if (i+1)*batch_size > len(list_data):
                data = list_data[i*batch_size:(i+1)*batch_size]
            else:
                data = list_data[i*batch_size:len(list_data)]
            data = torch.stack(data)
            data = data.to(self.device)
            out = self.model(data)
            torch.cuda.empty_cache()
            # find max
            _, index = torch.max(out, 1)
            # find prob use activation softmax
            percentage = (nn.functional.softmax(out, dim=1) * 100).tolist()
            for idx, name in enumerate(index):
                label.append(self.class_names[name])
                score.append(max(percentage[idx]))
        return label, score


if __name__ == '__main__':
    import random
    from glob import glob
    import time




