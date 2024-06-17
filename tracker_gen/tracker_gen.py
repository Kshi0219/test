from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import pandas as pd
import torch
import sys
sys.path.append('../')
from utils import *
from player_ball_assigner import *

class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.tracker=sv.ByteTrack(lost_track_buffer=150)
    
    def interpolate_ball(self,bal_detections):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in bal_detections]
        df_ball_positions = pd.DataFrame(ball_positions,
                                         columns=['x1','y1','x2','y2'])

        df_ball_positions_itp = df_ball_positions.interpolate()
        df_ball_positions_bf=df_ball_positions_itp.bfill()

        ball_positions_bf = [
            {1: {"bbox":x}} for x in df_ball_positions_bf.to_numpy().tolist()
            ]
        return ball_positions_bf
    
    def detect_frames(self,frames):
        batch_size=30
        detections=[]
        for i in range(0,len(frames),batch_size):
            track_result=self.model.predict(frames[i:i+batch_size],conf=0.55)
            detections+=track_result
        return detections
    
    def tracks_generator(self,frames,read_stub=False,stub_path=None):
        if read_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as l:
                tracks=pickle.load(l)
            return tracks
        
        detections=self.detect_frames(frames)
        tracks={'players':[],   #[{track_id:{'bbox:[x1,y1,x2,y2]}}]
                'ball':[]
                }
        for frame_num,detection in enumerate(detections):
            class_name=detection.names  # {class_id : class_name}
            class_inv={cls_name:cls_id for cls_id,cls_name in class_name.items()}   # {class_name : class_id}
            detection_sv=sv.Detections.from_ultralytics(detection)
            detection_with_tracks=self.tracker.update_with_detections(detection_sv)
            tracks['players'].append({})
            tracks['ball'].append({})
            # 바운딩 박스가 음수일 경우 0으로 대체
            for frame_detection in detection_with_tracks:
                bounding_box=frame_detection[0].tolist()
                for i in range(len(bounding_box)):
                    if bounding_box[i]<0:
                        bounding_box[i]=0
                class_id=frame_detection[3]
                track_id=frame_detection[4]
                if class_id==class_inv['player']:
                    tracks['players'][frame_num][track_id]={'bbox':bounding_box}
            for frame_detection in detection_sv:
                bounding_box=frame_detection[0].tolist()
                class_id=frame_detection[3]
                if class_id==class_inv['ball']:
                    tracks['ball'][frame_num][1]={'bbox':bounding_box}
        # self.test=frame_detection[2] - confidence
        if stub_path is not None:
            with open(stub_path,'wb') as s:
                pickle.dump(tracks,s)
        return tracks