import numpy as np
import cv2
import sys
sys.path.append('.../')
from utils import get_bbox_width,get_center_of_bbox

class annotator:
    def __init__(self):
        pass
    # keypoint에 원 그리기
    def draw_circle(self,frame,bbox,class_name,coord,conf):
        x1,y1=get_center_of_bbox(bbox)
        keypoint=(x1,y1)
        cv2.circle(
            frame,
            keypoint,
            3,
            (0,0,0),
            1,
            cv2.LINE_4
        )
        cv2.putText(
            frame,
            f"{class_name} : {coord}",
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            1
        )
        cv2.putText(
            frame,
            f"conf : {conf}",
            (x1,y1+10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            1
        )
        return frame

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2=int(bbox[3])-5
        x_center,_=get_center_of_bbox(bbox)
        width=get_bbox_width(bbox)

        # 선수, 심판 밑에 ellipse 그리기
        cv2.ellipse(
                frame,
                center=(x_center,y2),
                axes=(int(width*0.7),int(width*0.2)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=2,
                lineType=cv2.LINE_4
            )

        # 선수 하단 ellipse 중간에 track id 표시
        # 사각형 안에 track id 표시
        rectangle_width=20
        rectangle_height=10
        # 바운딩 박스와 같이 xyxy형태로 사각형 포인트 설정
        x1_rect=x_center-rectangle_width//2
        x2_rect=(x_center+rectangle_width//2)+10
        y1_rect=(y2-rectangle_height//2)+10
        y2_rect=(y2+rectangle_height//2)+10
        if track_id is not None:
            cv2.rectangle(
                    frame,
                    (int(x1_rect),int(y1_rect)),
                    (int(x2_rect),int(y2_rect)),
                    color,
                    cv2.FILLED
                )

            x1_text=x1_rect+6
            y1_text=y1_rect+10
            cv2.putText(
                    frame,
                    f"{track_id}",
                    (int(x1_text),int(y1_text)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,0,0),
                    1
                )
        return frame
        
    # 공 상단에 삼각형 그리기
    def draw_triangle(self,frame,bbox,color):
        x,_=get_center_of_bbox(bbox)
        y=int(bbox[1])
        triangle_points=np.array([
                [x,y],
                [x-5,y-10],
                [x+5,y-10]
            ])
        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        # 검은색 테두리 추가
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),1)
        return frame
        
    def annotate(self,frames,tracks,keypoints=None):
        output_frames=[]
        for frame_num,frame in enumerate(frames):
            frame=frame.copy()
            # keypoint_detection=keypoints[frame_num]
            player_dict=tracks['players'][frame_num]
            ball_dict=tracks['ball'][frame_num]

            '''# keypoint annotate - track id 없음
            for i in keypoint_detection:
                for keypoint_name,values in i.items():
                    bbox=values['bbox']
                    coord=values['coordinate']
                    conf=values['conf']
                    frame=self.draw_circle(frame,bbox,keypoint_name,coord,conf)'''

            # 필드플레이어 annotate
            for track_id,player in player_dict.items():
                player_color_hsv=player.get('team_color',(0,0,255))
                player_color_rgb=cv2.cvtColor(np.array([[[i for i in player_color_hsv]]],dtype=np.uint8),cv2.COLOR_HSV2BGR_FULL).tolist()[0][0]
                frame=self.draw_ellipse(frame,player['bbox'],tuple(player_color_rgb),track_id)
                if player.get('has_ball',False):
                    frame=self.draw_triangle(frame,player['bbox'],(0,0,255))

                
            # 공 annotate - track id 불필요
            for _,ball in ball_dict.items():
                ball_color=(0,255,0)
                frame=self.draw_triangle(frame,ball['bbox'],ball_color)
                
            output_frames.append(frame)
        return output_frames