# ACTION RECOGNITION WITH ST-GCN OR LSTM

## Introduction
Falls are a very common unexpected accident in the elderly that result in serious injuries such as broken bones, head injury. Detecting falls and taking fall patients to the emergency room in time is very important. In this project, we propose a method that combines face recognition and action recognition for fall detection. Specifically, we identify seven basic actions that take place in elderly daily life based on skeleton data detected using YOLOv7-Pose model. Two deep models which are Spatial Temporal Graph Convolutional Network (ST-GCN) and Long Short-Term Memory (LSTM) are employed for action recognition on the skeleton data. The experimental results on our dataset show that ST-GCN model achieved an accuracy of 90% higher than the LSTM model by 7%. 
## video demo
https://user-images.githubusercontent.com/87271954/204276637-f5d343de-9b19-43e4-a34b-5ffb7b696d9a.mp4
## System Diagram

![](./information/System_diagram.png)
## App
![](./information/app.png)
## Dev
```
Member:
 - DAO DUY NGU
 - LE VAN THIEN
Instructor: TRAN THI MINH HANH
```
## Usage
### Install package
```
git clone https://github.com/DuyNguDao/Human_Action_LSTM.git
```
```
cd Human_Action_LSTM
```
```
conda create --name human_action python=3.8
```
```
pip install -r requirements.txt
```
### Download
model yolov7 pose state dict:
[yolov7_w6_pose](https://drive.google.com/file/d/1UiDdOghLoRUOLbgkh41538oEXSG4dDUh/view?usp=share_link)

### Quick start
#### start with terminal
```
python detect_video.py --fn <url if is video or 0>
```
