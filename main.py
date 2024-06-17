import pickle
from utils import read_video,save_video
from tracker_gen import Tracker
from keypoint_detection import *
from draw_annotation import *
from team_assingn import *
from player_ball_assigner import *
import cv2

def main():
    # 영상 불러오기
    frames=read_video('input-video/test-adios-input.mp4')

    '''# keypoint detector 실행
    kpd_detector=keypointDetector('model/best_v5m_kpd_4.pt')
    keypoint_stub_path='kpd_stub/stub_1/stub-1-v5m-4-3(kp+coord).pkl'
    kpd_detection=kpd_detector.kpd_pickle_generator(frames,
                                                    read_stub=True,
                                                    stub_path=keypoint_stub_path)'''

    # 트래커 클래스 시작
    tracker=Tracker('model/test-best-v5m-2.pt')
    track_stub_path='track-stub/test-stub-v5-2-1.pkl'
    # 트래킹 결과 생성
    tracks=tracker.tracks_generator(frames,
                                    read_stub=True,
                                    stub_path=track_stub_path)
    
    # interpolate missing ball postion
    tracks['ball']=tracker.interpolate_ball(tracks['ball'])
    
    # 색상 생성
    tracks_assigned=TeamAssigner().add_2_tracks(frames,tracks)

    # assign ball aquisition
    tracks_assigned_2=ballAssigner().add_2_tracks(tracks_assigned)
    
    # 팀 정보 포함된 stub 저장
    with open(track_stub_path,'wb') as s:
        pickle.dump(tracks_assigned_2,s)

    # annotate
    output_frames=annotator().annotate(frames,tracks_assigned_2)


    # annotate한 영상 저장
    save_video(output_frames,
               'output-video/test-output-v5-2-1.avi')

if __name__=='__main__':
    main()