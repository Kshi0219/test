import sys
sys.path.append('../')
from utils import *

class ballAssigner():
    def __init__(self):
        self.max_distance=55
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position=get_center_of_bbox(ball_bbox)
        min_distance=99999
        assigned_player=-1
        # dist_list=[]
        for player_id,player in players.items():
            player_bbox=player['bbox']
            player_width=get_bbox_width(player_bbox)
            # distance=measure_distance(player_postion,ball_position)
            # dist_list.append(distance)
            # distance=min(dist_list)
            distane_left=measure_distance((player_bbox[0]+player_width*(4/10),
                                           player_bbox[-1]),ball_position)
            distane_right=measure_distance((player_bbox[2]-player_width*(4/10),
                                            player_bbox[-1]),ball_position)
            distance=min(distane_left,distane_right)

            if distance<self.max_distance:
                if distance<min_distance:
                    min_distance=distance
                    assigned_player=player_id
        return assigned_player
    
    def add_2_tracks(self,tracks):
        for frame_num,player_track in enumerate(tracks['players']):
            ball_bbox=tracks['ball'][frame_num][1]['bbox']
            assigned_player=self.assign_ball_to_player(player_track,ball_bbox)
            if assigned_player!=-1:
                tracks['players'][frame_num][assigned_player]['has_ball']=True
        return tracks