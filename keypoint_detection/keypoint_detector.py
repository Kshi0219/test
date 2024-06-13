from ultralytics import YOLO
import torch
import numpy as np
import supervision as sv
import pickle
import json
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox

class keypointDetector:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.tracker=sv.ByteTrack()
    
    def detect_keypoint(self,frames):
        detections=[]
        for frame in frames:
            result=self.model.predict(frame,conf=0.4)
            detections+=result
        return detections
    
    def kpd_pickle_generator(self,frames,read_stub=False,stub_path=None):
        if read_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as ld:
                kpd_detections=pickle.load(ld)
            return kpd_detections
        
        with open('keypoint_coordinate.json','r') as load:
            keypoint_coordinate=json.load(load)
        detections=self.detect_keypoint(frames)
        kpd_detections={}
        for frame_num,frame_detection in enumerate(detections):
            kpd_detections[frame_num]=[]
            detection_sv=sv.Detections.from_ultralytics(frame_detection)
            detection_track=self.tracker.update_with_detections(detection_sv)
            for detection in detection_track:
                bounding_box=detection[0].tolist()
                x,_=get_center_of_bbox(bounding_box)
                y=bounding_box[3]
                coordinate=(x,y)
                conf=round(detection[2],3)
                cls_name=self.model.names[detection[3]]
                keypoint=keypoint_coordinate[cls_name]
                kpd_detections[frame_num].append(
                    {cls_name:{'keypoint':keypoint,
                               'coordinate':coordinate,
                               'bbox':bounding_box,
                               'conf':conf}})
        
        if stub_path is not None:
            with open(stub_path,'wb') as s:
                pickle.dump(kpd_detections,s)
        return kpd_detections