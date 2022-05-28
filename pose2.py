import cv2
import mediapipe as mp
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt
from suppliments import class_handler as ch
import pickle
import time

import pyttsx3
engine = pyttsx3.init() # object creation
rate = engine.getProperty('rate')
engine.setProperty(rate, 175)
voiceinourpc=engine.getProperty('voices')
engine.setProperty('voice',voiceinourpc[1].id)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
Rotation=["clockwise","anticlockwise",""]
color_scale=[(0,0,255),(0,255,255),(0,255,0)]

sink_origin="Trikon/yoga1.mp4"

sink = cv2.VideoCapture(0)
source = cv2.VideoCapture(sink_origin)
source_fps=source.get(cv2.CAP_PROP_FPS)
width  = source.get(3)   # float `width`
height = source.get(4)
sink.set(cv2.CAP_PROP_POS_FRAMES, 10)

## create objects for all the important joint, using the images from data/base.png

l_elbow=ch.Angles(" your left elbow",[11,13,15])
r_elbow=ch.Angles(" your right elbow",[12,14,16])

l_shoulder=ch.Angles(" your left shoulder",[13,11,23])
r_shoulder=ch.Angles(" your right shoulder",[14,12,24])

l_knee=ch.Angles(" your left knee",[23,25,27])
r_knee=ch.Angles(" your right knee",[24,26,28])

l_hip=ch.Angles(" your left hip",[11,23,25])
r_hip=ch.Angles(" your right hip",[12,24,26])

### Add all the objects into this list

body_parts=[l_elbow,r_elbow,l_shoulder,r_shoulder,l_hip,r_hip,l_knee,r_knee]

### create the evaluation function, for eg: [trikon left_hip_angle]-[right_hip_angle]

def trigger_func(left_hip, right_hip):
    ## parameters should be from the list of objects
    return left_hip-right_hip
    

trik_asana=ch.Asanas(asana_name="Trikon",variables=[l_hip,r_hip],triggers={"left":{"g":30},"right":{"l":-30}},function=trigger_func)



def align_pose(sink_results,source_results):
    global height,width
    angles=[]
    base=np.array([1,0])
    shoulder_local=[]
    hip_local=[]
    for i in [sink_results,source_results]:
        x_mid=0.50*(i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x+i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        y_mid=0.50*(i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y+i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x=x_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y=y_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x=x_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y=y_mid
        shoulder_local.append(np.array([x_mid,y_mid]))

        x_mid=0.50*(i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x+i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x)
        y_mid=0.50*(i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y+i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x=x_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y=y_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x=x_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y=y_mid
        hip_local.append(np.array([x_mid,y_mid]))
        
    sink_vector=[sink_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x-sink_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                sink_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y-sink_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y]
    source_vector=[source_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x-source_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                source_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y-source_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y]
    scale=1
  

    offset=hip_local[0]-hip_local[1]

    for landmark_name in mp_pose.PoseLandmark:
        point=np.array([source_results.pose_landmarks.landmark[landmark_name].x,source_results.pose_landmarks.landmark[landmark_name].y])
        new_point=(point-hip_local[1])*scale+hip_local[1]+offset
        source_results.pose_landmarks.landmark[landmark_name].x=new_point[0]
        source_results.pose_landmarks.landmark[landmark_name].y=new_point[1]

def track_all_angles(source_pose,sink_pose,body_parts=body_parts):
    for tag,pose in zip(["source","sink"],[source_pose,sink_pose]):
        for part in body_parts:
            part.track_angle(pose,tag)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

befor = time.time()
def display_instruction(stage,body_parts=body_parts):
    global befor
    if stage=="right":
        file_name="Trikon/IDEAL/trik_ideal_right.pickle"
    else:
        file_name="Trikon/IDEAL/trik_ideal.pickle"

    load_file=open(file_name,"rb")
    ideal=pickle.load(load_file)
    load_file.close()

    for n in range(len(body_parts)):
        text=""
        match,performance=body_parts[n].generate_instructions(ideal[n])
        if performance>=50:
            color=color_scale[0]
            text+="bend"
        elif performance>=25:
            color=color_scale[1]
            text+="try bending a bit "
        else:
            color=color_scale[2]
        if match==2:
            text+="looks good "   
        else:
            pass
       
    
        cv2.putText(sink_image, text + " " + body_parts[n].name+": "+ Rotation[match],(60,30+n*30),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        if performance >= 25:
            if time.time() - befor >= 3:
                speak(text + " " + body_parts[n].name + " " + Rotation[match])
                befor = time.time()
    cv2.putText(sink_image,"Stage:" + str(stage),(800,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

sink_pose=mp_pose.Pose(
    min_detection_confidence=0.5,
    # model_complexity=1,
    min_tracking_confidence=0.5) 
source_pose=mp_pose.Pose(
    min_detection_confidence=0.5,
    # model_complexity=1,
    min_tracking_confidence=0.5) 

frame=0

while sink.isOpened(): 
    success, sink_image = sink.read()
    # source_image= cv2.imread("Data/pose.jpg")
    _,source_image= source.read()
    if not success or not _:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    frame+=1
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    sink_image = cv2.cvtColor(sink_image, cv2.COLOR_BGR2RGB)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    # source_image = cv2.cvtColor(cv2.flip(source_image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    sink_image.flags.writeable = False
    source_image.flags.writeable = False
    sink_results = sink_pose.process(sink_image)
    source_results = source_pose.process(source_image)
    cv2.flip(sink_image, 1)
    try:
        track_all_angles(source_results,sink_results)
    except:
        pass
    
    # Draw the pose annotation on the image.
    sink_image.flags.writeable = True
    sink_image = cv2.cvtColor(sink_image, cv2.COLOR_RGB2BGR)
    try:
        align_pose(sink_results,source_results)
    except Exception as e:
        print("super impose:",e)

    mp_drawing.draw_landmarks(
        sink_image, sink_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(),mp_drawing.DrawingSpec(color=(0,0,255)))

    mp_drawing.draw_landmarks(
        sink_image, source_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


    try:
        stage=trik_asana.detect_stage()
        display_instruction(stage)
    except Exception as e:
        print("stage detection + istruction generation:",e)
    
    sink_image = cv2.resize(sink_image, (720, 480)) 
    cv2.imshow("sink",sink_image)
    source_image = cv2.resize(source_image, (720, 480)) 
    cv2.imshow("source",source_image)


    


    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(-1)
    elif key == ord('d'):
        frame+=60
        sink.set(cv2.CAP_PROP_POS_FRAMES, frame)
        source.set(cv2.CAP_PROP_POS_FRAMES, frame)
    elif key == ord('a'):
        frame-=60
        sink.set(cv2.CAP_PROP_POS_FRAMES, frame)
        source.set(cv2.CAP_PROP_POS_FRAMES, frame)
        

    

sink.release()
source.release()


scores=[]
for part in body_parts:
    score=part.generate_score()
    scores.append(score)
    print(part.name,score)

print("average score = ",np.average(np.array(scores)))
