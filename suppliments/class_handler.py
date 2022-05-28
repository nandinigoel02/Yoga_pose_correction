import numpy as np
import mediapipe as mp
from dtaidistance import dtw

mp_pose=mp.solutions.pose
landmarks_name=[]
Threshhold=25

for i in mp_pose.PoseLandmark:
    landmarks_name.append(i)

# print(landmarks_name)

def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

class Angles:
    def __init__(self, joint_name="to be set",points="to be set"):
        self.name=joint_name
        self.points=points
        self.source_tracker=[]
        self.sink_tracker=[]
        self.score=None
        self.threshold=Threshhold
        self.match=1

        # print("tracking "+str(self.name)+", tracking part "+str(landmarks_name[points[0]])+str(landmarks_name[points[1]])+str(landmarks_name[points[2]]))

    def calculate_angle(self,pose,tag="source"):
        if self.points=="to be set" or len(self.points)!=3:
            print(self.name,"the points have not been set properly")
            return
        a=np.array([pose.pose_world_landmarks.landmark[landmarks_name[self.points[0]]].x,pose.pose_world_landmarks.landmark[landmarks_name[self.points[0]]].y]) 
        b=np.array([pose.pose_world_landmarks.landmark[landmarks_name[self.points[1]]].x,pose.pose_world_landmarks.landmark[landmarks_name[self.points[1]]].y]) 
        c=np.array([pose.pose_world_landmarks.landmark[landmarks_name[self.points[2]]].x,pose.pose_world_landmarks.landmark[landmarks_name[self.points[2]]].y]) 
        # print(a,b,c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def track_angle(self,pose,tag="source"):
        # print(landmarks_name[23],pose.pose_landmarks.landmark[landmarks_name[23]].visibility ,pose.pose_landmarks.landmark[landmarks_name[23]].presence )
        angle=self.calculate_angle(pose)
        # print(angle)
        if tag=="source":
            self.source_tracker.append(angle)
        else:
            self.sink_tracker.append(angle)
        

    def generate_score(self):
        sink_actions,source_actions=np.array(self.sink_tracker)/180,np.array(self.source_tracker)/180
        sink_actions=sink_actions/np.linalg.norm(sink_actions)
        source_actions=source_actions/np.linalg.norm(source_actions)
        self.score=100*(1-dtw.distance(sink_actions,source_actions))
        return self.score

    def generate_instructions(self,ideal):
        angle1=self.sink_tracker[-1]
        angle2=ideal.source_tracker[-1]
        if angle1<angle2+self.threshold and angle1>angle2-self.threshold:
            self.match=2
        elif angle1>angle2+self.threshold:
            self.match=0
        elif angle1<angle2-self.threshold:
            self.match=1
        return self.match, np.abs(angle2-angle1)


class Asanas:
    def __init__(self, asana_name="to be set", variables=None, triggers=None, function=None):
        self.name=asana_name
        self.variables=variables
        self.triggers=triggers
        self.function=function

        if variables==None:
            print("provide variables")
        if triggers==None:
            print("provide triggers")
        if function==None:
            print("provide function")


    def detect_stage(self):
        vars=[x.sink_tracker[-1] for x in self.variables]
        vars=tuple(vars)
        output=self.function(*vars)

        for key in self.triggers.keys():
            trigger_val=list(self.triggers[key].values())[0]
            if "g" in self.triggers[key].keys():
                if output >= trigger_val:
                    return key
            else:
                if output <= trigger_val:
                    return key

        return -1
