import streamlit as st
from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd

import cv2
import mediapipe as mp
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns  
from mpl_toolkits.mplot3d import Axes3D
from tempfile import NamedTemporaryFile
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import cv2
import mediapipe as mp
import os 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
import seaborn as sns  
from mpl_toolkits.mplot3d import Axes3D # 3D 시각화
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from pywt import wavedec
from imutils.video import VideoStream
import f_detector #만든 detector 라이브러리 
import imutils 



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(layout="wide", page_title="Tic Analysis")
st.write("## Psycho Detector : 청소년기 정신장애분류 AI ")
st.write("BME MIP Project 2023 : 최대현, 정영진, 이수하, 이상경")
st.sidebar.write("## Upload the Video")
plt.rcParams['font.size']=3

## 얼굴 외곽 마스크 생성 방법
def face_outline_masking(results, landmark_dict, image, label):
    # 얼굴 외곽 정의한 것 가지고 얼굴 외곽 좌표 추출
    face_outline_coords = [
        (int(results.multi_face_landmarks[0].landmark[i].x * image.shape[1]),
        int(results.multi_face_landmarks[0].landmark[i].y * image.shape[0]))
        for i in landmark_dict['Outline_face']] 

    # 얼굴 외곽 마스크 생성
    mask = np.zeros_like(image)
    
    # 얼굴 부분을 빨간색으로 만들기
    cv2.fillPoly(mask, [np.array(face_outline_coords, dtype=np.int32)], (0, 0, 255))  # R : BGR 순서
    # cv2.fillPoly(mask, [np.array(face_outline_coords, dtype=np.int32)], (0, 255, 0))  # G : BGR 순서
    # cv2.fillPoly(mask, [np.array(face_outline_coords, dtype=np.int32)], (255, 0, 0))  # B : BGR 순서
    
    # 얼굴 마스크 적용
    masked_face = cv2.bitwise_and(image, mask)
    # 원본 영상에 얼굴 마스크 덧씌우기
    masked_image = cv2.addWeighted(image, 1, masked_face, 0.5, 0.1)  
    # 얼굴 영역 좌표 구하기
    x, y, w, h = cv2.boundingRect(np.array(face_outline_coords, dtype=np.int32))
    # 얼굴 내부 부분에 박스 그리기
    cv2.rectangle(masked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 얼굴에 bounding box 이름 추가
    text_size = cv2.getTextSize('Face', cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)[0]
    # cv2.putText(masked_image, 'Face', (x + (w - text_size[0]) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 얼굴 마스크 적용
    masked_face = cv2.bitwise_and(masked_image, mask)
    # 원본 영상에 얼굴 마스크 덧씌우기
    masked_image = cv2.addWeighted(masked_image, 1, masked_face, 0.8, 0.1)

        
        # assign labeling
    cv2.putText(masked_image, f'Face Emotion : {label}', (x + (w - text_size[0]) // 2 - 70, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    return masked_face, masked_image, x,y, x+w, y+h, 

################################################################################################################################################
## hanging_cnt_v1 : z-score normal 후 부호를 이용해서 머리흔들림이 심한 프레임에 대해서 부호로 머리흔들림 스코어링 선정 -> 잘안될떄가 있다.
def hanging_cnt_v1(hanging_frame):
    #움직임에 대한 변화량을 z_score_normalization 함
    movement_array = z_score_norm(np.array(hanging_frame)[:,1])
    # print(f"sign change check for z_score_movement : {np.sign(movement_array)}")
    # movement_array = z_score_norm(hanging_frame[:,1])

    # 부호 변경 횟수를 카운트할 변수
    sign_change_count = 0

    # 이전 움직임의 부호 초기화
    previous_sign = np.sign(movement_array[0])

    # 배열을 순회하면서 부호 변경 횟수를 계산
    for i in range(1, len(movement_array)):
        current_sign = np.sign(movement_array[i])
        
        # 부호가 변경되면 카운트 증가
        if current_sign != previous_sign:
            # 다음 움직임이 다시 원래 부호로 돌아올 때만 카운트 증가
            if i + 1 < len(movement_array) and np.sign(movement_array[i + 1]) == previous_sign:
                # print(hanging_frame[])
                sign_change_count += 1
        
        previous_sign = current_sign

    print(f"v1_틱 증상  : {sign_change_count}")
################################################################################################################################################
## hanging_cnt_v2 : 프레임의 대한 범위를 지정해서 하나의 그룹으로 묶는다. 단점 : 프레임 범위인 threshold를 정해줘야함.
def hanging_cnt_v2(frames, threshold):
    # 그룹을 저장할 리스트
    groups = []

    # 현재 그룹에 속하는 프레임들을 저장할 리스트
    current_group = [frames[0]]

    # 프레임별로 그룹을 묶음
    for i in range(1, len(frames)):
        # 현재 프레임과 이전 프레임의 차이가 10프레임을 넘지 않으면 같은 그룹으로 묶음
        if frames[i] - frames[i-1] <= threshold:
            current_group.append(frames[i])
        else:
            # 차이가 10프레임을 넘으면 새로운 그룹 시작
            groups.append(current_group)
            current_group = [frames[i]]

    # 마지막 그룹 추가
    groups.append(current_group)
    # 결과 출력
    return len(groups)

################################################################################################################################################

def is_eye_closed(results, landmark_dict, image, threshold, tag):
    # 눈 가/세로 높이 정하기 위한 랜드마크 정의한 것 가지고 눈 좌표 추출
    # eye_blink_coords = np.array([
    #     (results.multi_face_landmarks[0].landmark[i].x,
    #     results.multi_face_landmarks[0].landmark[i].y)
    #     for i in landmark_dict[tag +'_eye_blink']])
    eye_blink_coords = np.array([
        (int(results.multi_face_landmarks[0].landmark[i].x * image.shape[1]),
        int(results.multi_face_landmarks[0].landmark[i].y * image.shape[0]))
        for i in landmark_dict[tag +'_eye_blink']])    
    
    # 눈 중심좌표 계산 
    eye_center = np.mean(eye_blink_coords, axis = 0)
    
    # 눈의 높이와 폭 계산
    eye_height = np.linalg.norm(eye_blink_coords[1] - eye_blink_coords[3])
    eye_width = np.linalg.norm(eye_blink_coords[0] - eye_blink_coords[2])

    # # 눈의 종횡비 계산
    aspect_ratio = eye_width / eye_height
    
    # # 눈이 감겨 있는지 여부를 판단
    # 임의의 임계값, 실험을 통해 조절
    if aspect_ratio > threshold:
        return 1, aspect_ratio
    else:
        return 0, aspect_ratio

################################################################################################################################################

#영역 부피 저장 함수
def save_area(area_array, tag, video_info):
    np.save(os.path.join('./data/np', str(video_info['path'].split('/')[-1].split('.')[0] + f'_{tag}')),area_array)

    
def z_score_norm(arr):
    # z-score normalization
    mean_val = np.mean(arr)
    std_dev = np.std(arr)
    return (arr - mean_val) / std_dev

# 전체 histogram 
def total_weight_score(hanging_score, unpair_blink_score, blink_score, emotion_score, w1, w2, w3, w4):
    total = w1 * hanging_score + w2 * unpair_blink_score + w3 * blink_score + w4 * emotion_score
    caused_tic_percent = total /(w1+w2+w3+w4)
    # 틱 장애 위험도 
    return caused_tic_percent 

#머리를 흔드는 것에 대한 점수화 
def hanging_face_score(cnt, play_time, threshold):
    score = (cnt/play_time) * threshold
    if score < 0.5:
        return 0
    elif score > 0.5 and score < 1.0:
        return 0.2
    elif score > 1.0 and score < 1.5:
        return 0.4
    elif score > 1.5 and score < 2.0:
        return 0.6
    elif score > 2.0 and score < 2.5:
        return 0.8
    else:
        return 1.0

## 비대칭적으로 눈 감는 횟수를 점수화
def unpair_eye_blink_score(left_cnt, right_cnt, threshold):
    # 0~1 사이로 정규화
    abs_cnt = abs(left_cnt - right_cnt)
    if abs_cnt // threshold == 0: #기준치보다 미달일경우 0점 할당 
        return 0
    else: #0.2, 0.4, 0.6, 0.8, 1.0
        if abs_cnt // threshold == 1: 
            return 0.2
        elif abs_cnt // threshold == 2: 
            return 0.4
        elif abs_cnt // threshold == 3: 
            return 0.6
        elif abs_cnt // threshold == 4: 
            return 0.8
        else:
            return 1.0
        
## 평균적으로 1분에 10~15회 즉 5초에 1회정도 눈을 감는다를 정의 
def blink_equal_score(left_cnt, right_cnt, play_time, threshold):
    equal_cnt = ((left_cnt + right_cnt)/2) // play_time * threshold #threhold 시간당 1번 감는 평균 횟수 되는거임
    if equal_cnt < 1:
        return 0
    elif equal_cnt == 2:
        return 0.2
    elif equal_cnt == 3:
        return 0.4
    elif equal_cnt == 4:
        return 0.6
    elif equal_cnt == 5:
        return 0.8
    else:
        return 1.0
    
## upload하는 영상을 웹에 출력하는 함수
def upload_video(upload):
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    # st.write(temp_file)
    # st.write(temp_file.name)
    # st.write(uploaded_file.name)

    cap = cv2.VideoCapture(temp_file.name)

    change_format = uploaded_file.name[:-4] + '.webm'
    # st.write(change_format)
    
    landmark,blink_frame,video_info,hanging_frame,res_image,results,emotion_x,emotion_y = process_data(temp_file.name,uploaded_file.name)

    masked_path = rf"./streamlit_result/masked2__{change_format}"
    # st.write(masked_path)
    
    
    col1.write('### Original Video')
    col1.video(temp_file.name)
    
    col2.write('### Face Mesh Detection Video')
    col2.video(masked_path, format = 'video/webm')
    
    return landmark,blink_frame,video_info,hanging_frame,res_image,results,emotion_x,emotion_y


## 얼굴 감정 감지 방법
def face_emotion_detector(results, landmark_dict, image, emotion_classifier, EMOTIONS):
    # 얼굴 외곽 정의한 것 가지고 얼굴 외곽 좌표 추출
    face_outline_coords = [
        (int(results.multi_face_landmarks[0].landmark[i].x * image.shape[1]),
        int(results.multi_face_landmarks[0].landmark[i].y * image.shape[0]))
        for i in landmark_dict['Outline_face']] 
    x, y, w, h = cv2.boundingRect(np.array(face_outline_coords, dtype=np.int32))
    
    ## 감정 인식
    # resize image 48*48 for neural network
    # 얼굴 부분을 원본 이미지 그대로 만들기
    # face = np.array(face_outline_coords, dtype=np.int32)
    # face = cv2.fillPoly(mask, [np.array(face_outline_coords, dtype=np.int32)], (255, 255, 255))  # 흰색
    roi = image[y:y+h,x:x+w] 
    # print(roi)
    # plt.imshow(roi)
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = np.array(roi, dtype = np.int32)/255.0
    roi = img_to_array(roi)
    roi = cv2.resize(roi, (64,64))
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    # print(roi.shape)
    # emothoin predict
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    # print(emotion_probability)
    # print(preds)
    label = EMOTIONS[preds.argmax()]
    # print(label)
    return label
   
## 얼굴 변화 감지 스코어링 
def emotion_scoring(emotion_array):
    emotion_dict = {}
    for emotion in emotion_array:
        emotion_dict[emotion] = 0
    for idx, emotion in enumerate(emotion_array):
        emotion_dict[emotion] += 1
    emotion_x = list(emotion_dict.keys())
    emotion_y = np.array(list(map(float, list(emotion_dict.values()))))
    
    # plt.bar(emotion_x, emotion_y)
    # plt.tight_layout()
    # plt.title('Face Expression Analysis')
    # plt.xlabel('Face Expression Range')
    # plt.ylabel('Face Expression Count')
    # # plt.show()
    # # plt.savefig('./analysis/emotion_scoring.png')
    # plt.close()
    
    ## scoring 방법 : 보통 공부할때 neutral이 나오는 경우가 많으므로 그 외의 경우 normalization시켜서 감점시킨다.
    score = (np.sum(emotion_y)- emotion_dict['Neutral']) / np.sum(emotion_y)
    return score,emotion_x,emotion_y 

def emotion_bar(emotion_x,emotion_y):
    
    col5_1.write('###### 감정표현 예측 결과 분석 그래프')
    fig = plt.figure()
    plt.bar(emotion_x, emotion_y)
    plt.tight_layout()
    plt.title('Face Expression Analysis')
    plt.xlabel('Face Expression Range')
    plt.ylabel('Face Expression Count')
    col5_1.pyplot(fig)
    # plt.show()
    
       

## Mask를 입힌 동영상을 생성하고 저장하는 코드
def process_data(video_data,name): 
    landmark_dict = {
        #왼쪽눈
        'L_eye' : [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7], #시계방향 9시부터 시작
        #오른쪽눈
        'R_eye' : [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382], #시계방향 9시부터 시작
        #왼쪽 눈동자
        'L_iris' : [470, 471,469, 472, 468], #위에서부터 Up, Left, Right, Down, Center position
        #오른쪽 눈동자
        'R_iris' : [475, 476, 474, 477, 473],
        #왼쪽 눈썹
        'L_eyebrow' : [70, 63, 105, 66, 107, 55, 65, 52 ,53, 46], #시계방향 9시부터 시작
        #오른쪽 눈섭
        'R_eyebrow' : [336, 296, 334, 293, 300, 276, 283, 282, 295 ,285], #시계방향 9시부터 시작
        #윗 입술
        'Up_lips' : [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 391, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],#시계방향 9시부터 시작
        #아랫 입술
        'Down_lips' : [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 391, 375, 321, 405, 314, 17, 84, 181, 91, 146],#시계방향 9시부터 시작
        #입술 안
        'Inner_lips' : [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],#시계방향 9시부터 시작
        #얼굴외곽
        'Outline_face' : 
            [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284,
            251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93],
        'L_eye_blink' : [33, 159, 133, 145],
        'R_eye_blink' : [362, 386, 263, 374], 
    }
    
    SAVE_PATH = r'./streamlit_result' #project Folder/result/maksed_{pid}.mp4 로 저장 됨
    cap = cv2.VideoCapture(video_data) # video load 
    video_info = {
        'name' : video_data.split('/')[-1][:-4]+".webm",
        'path' : video_data,
        'fps' : cap.get(cv2.CAP_PROP_FPS),
        'total frame' : int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), # 프레임수
        'width' : int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), # 가로 길이
        'height' : int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), # 세로 길이
        "running_time('s)" : int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
    }
    out = cv2.VideoWriter(
        SAVE_PATH + '/masked2__' + name[:-4]+'.webm',
        cv2.VideoWriter_fourcc(*'vp80'),
        video_info['fps'],
        (video_info['width'], video_info['height'])
        )
    masked_path = SAVE_PATH + '/masked2__' + name[:-4]+'.webm'
    
    # print(f"Video Information : {video_info}")
    
    ## Video Setting
    # cap = cv2.VideoCapture(video_data) # video load 
    drawing_spec = mp_drawing.DrawingSpec(thickness = 3, circle_radius = 3)

    ## 얼굴 랜드마크 검출 객체
    landmark = []

    ## fps 탐지
    import time
    start_time = time.time()

    ## 머리 흔들기 객체
    hanging_cnt = 0
    hanging_threshold = 0.03
    hanging_frame = []

    ## 눈 감기는 횟수 객체 
    left_eye_closed_cnt = 0
    right_eye_closed_cnt = 0
    eye_blink_threshold = 5 # 눈 감길때 논의 종횡비 계산 threshold
    blink_frame = []

    ## face detection XML load and trained inference model loading
    face_detection = cv2.CascadeClassifier('./model/haarcascade_frontalface_transfer.xml')
    emotion_classifier = load_model('./model/emotion_model.hdf5', compile = False)
    EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]

    ## Emotion Array Save
    emotion_li = []
    
    ### Video Action Code
    with mp_face_mesh.FaceMesh(  
            max_num_faces=3, #최대 검출 얼굴 개수
            refine_landmarks=True, # 눈과 입술 주변 랜드마크 정교하게 검출시 True
            min_detection_confidence=0.5, #최소 Detection 기준
            min_tracking_confidence=0.5, #최소 traicking 기준
            static_image_mode=True) as face_mesh:
        frame = 1
        while cap.isOpened():
            # print(f"frame : {frame} frame")
            
            ret, image = cap.read()
            # 현재 시간과 이전 시간과의 경과 시간 계산
            elapse_time = time.time() - start_time
            
            # 현재 fps 계산
            fps = 1 / elapse_time
            
            # if not ret:
            #     print("웹캠을 찾을 수 없습니다.")
            #     # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요
            #     # continue
            #     break
            if not ret: #영상 더 가지고 올게 없으면 끝내기
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break
            else: #영상 진행 중일때 
                # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # 이미지 위에 얼굴 그물망 주석을 그립니다.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    #face detector            
                    label = face_emotion_detector(results, landmark_dict, image, emotion_classifier, EMOTIONS)
                    emotion_li.append(label)
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())
                        
                        ## masking 및 박스 플랏 이미지
                        masked_face, masked_image, x,y, x_w, y_h,  = face_outline_masking(results, landmark_dict, image, label)
                        
                        ## 눈 깜빡임 횟수 정의is_eye_closed
                        l_cnt, left_eye_landmarks = is_eye_closed(results, landmark_dict, image, eye_blink_threshold, tag = 'L')    
                        r_cnt, right_eye_landmarks = is_eye_closed(results, landmark_dict, image, eye_blink_threshold, tag = 'R')    
                        # 프레임 별로 저장
                        blink_frame.append([frame, left_eye_landmarks, right_eye_landmarks])
                        # 눈 깜빡임 횟수 추가
                        left_eye_closed_cnt += l_cnt
                        right_eye_closed_cnt += r_cnt

                        ## 프레임 구간별(default : 10fps) 얼굴 움직임 변화량 측정
                        # 현재 랜드마크 좌표
                        current_landmark = np.array([(point.x, point.y, point.z) for point in face_landmarks.landmark])
                        if frame == 1: #첫 번쨰 프레임일떄는
                            # 현재 랜드마크를 이전 랜드마크로만 저장
                            previous_landmark = current_landmark
                            tag_detected = False
                            tag = 0
                            distances = 0
                            continue
                        else:
                            # 모든 랜드마크의 변화량 계산
                            distances = np.linalg.norm(np.abs(current_landmark - previous_landmark), axis=1)
                            # 평균변화량 출력 : Heuristic threshold Setting 0.05  
                            if np.mean(distances) > hanging_threshold:
                                # print(f"Landmark Movement|frame :{np.mean(distances)}|{frame}")
                                hanging_frame.append([frame, np.mean(distances)])
                                tag_detected = True 
                                tag += 1
                            else:
                                hanging_frame.append([frame, 0])
                                tag_detected = False
                            # 현재 랜드마크를 이전 랜드마크로 저장
                            previous_landmark = current_landmark



                try:
                    landmark.append(results.multi_face_landmarks)        
                except TypeError:
                    continue
                
                ## FPS 표시
                try:
                    cv2.putText(masked_image, f"FPS : {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                except:
                    not_human = 'Not Human'
                    print('-----warning!!! : This video is not human -----------')
                    break
                ## frame 표시
                cv2.putText(masked_image, f"Frame : {frame}", (0, video_info['height']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA) 
                ## 초 표시
                second_record = (frame / video_info['fps'])
                cv2.putText(masked_image, f"Play time : {second_record:.2f}sec", (video_info['width']-180, video_info['height']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA) 
                            
                ## 눈이 감겼을 때 메시지 추가
                if l_cnt == 1: #눈 감겼을때 cnt = 1로 지정한 것으로 판단 x,y, x_w, y_h
                    cv2.putText(masked_image, f"R_Blink : {right_eye_landmarks/10:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # cv2.putText(masked_image, f"Blink {right_eye_closed_cnt}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(masked_image, f"R_Blink : {right_eye_landmarks/10:.2f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    # cv2.putText(masked_image, f"Blink : {right_eye_closed_cnt}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                if r_cnt == 1: #눈 감겼을때 cnt = 1로 지정한 것으로 판단
                    cv2.putText(masked_image, f"L_Blink : {left_eye_landmarks/10:.2f}", (x_w-70, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # cv2.putText(masked_image, f"Blink : {left_eye_closed_cnt}", (x_w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(masked_image, f"L_Blink : {left_eye_landmarks/10:.2f}", (x_w-70, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    # cv2.putText(masked_image, f"Blink : {left_eye_closed_cnt}", (x_w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                
                ## 얼굴 큰 흔들림 태그 감지 및 횟수 메시지 추가
                if tag_detected:
                    cv2.putText(masked_image, "Head hadly shake detect!!", ((x+x_w)//2, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(masked_image, f"Head Hanging Count : {tag} | {np.mean(distances*10):.2f}", (video_info['width']-250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) 
                else:
                    cv2.putText(masked_image, f"Head Hanging Count : {tag} | {np.mean(distances*10):.2f}", (video_info['width']-250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA) 
                
                # 보기 편하게 이미지를 좌우 반전합니다.
                # edge_image = cv2.Canny(image, 100, 255)
                
                # cv2.imshow('MediaPipe Face Mesh(Puleugo)', image)
                cv2.imshow('MediaPipe Face Mesh(Puleugo)', masked_image)
                # cv2.imshow('MediaPipe Face Mesh(Puleugo)', frame_image)
                
                # 현재 시간 갱신
                start_time = time.time()
                
                ## 동영상 파일 저장
                out.write(masked_image)
                res_image = image
                # Esc 누르면 영상 종료
                if cv2.waitKey(5) & 0xFF == 27:
                    #맥에서 opencv 안닫힐때 꿀팁
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    cv2.waitKey(1)
                    cv2.waitKey(1)
                    cv2.waitKey(1)
                    break
            frame += 1
    cap.release()
    out.release()
### Algorithms Result
# hanging_cnt_v1(hanging_frame)
    try:
        hanging_cnt = hanging_cnt_v2(np.array(hanging_frame)[:,0], threshold= video_info['fps'])
    except:
        hanging_cnt = 0
    col3.write("           ")
    col3.write("           ")
    col3.write("##### 틱 증상 횟수 측정")
    col3.write(f"틱 증상 횟수 : {hanging_cnt}번")
    col3.write(f"왼쪽 눈 감는 횟수 : {left_eye_closed_cnt}회 | 프레임 기준")
    col3.write(f"오른족 눈 감는 횟수 : {right_eye_closed_cnt}회 | 프레임 기준\n")
    col3.write("           ")
    col3.write("           ")

    ### Scoring System
    col4.write("           ")
    col4.write("           ")
    col4.write("##### 각 영역별 틱 점수")
    hanging_score = hanging_face_score(hanging_cnt, second_record, threshold = 5) #5초당 평균 머리 흔들기 횟수 점수화
    col4.write(f"평균 머리 흔들기 횟수 이상 정도 : {hanging_score * 100.}%")
    unpair_blink_score = unpair_eye_blink_score(left_eye_closed_cnt, right_eye_closed_cnt, threshold = 3)
    col4.write(f"비대칭 눈감기 이상 정도 : {unpair_blink_score * 100.}%")
    blink_score = blink_equal_score(left_eye_closed_cnt, right_eye_closed_cnt, second_record, threshold = 5)
    col4.write(f"평균 눈 감는 횟수 이상 정도 : {blink_score * 100.}%")
    emotion_score,emotion_x,emotion_y = emotion_scoring(np.array(emotion_li))
    col4.write(f"감정변화 점수 : {(emotion_score*100):.2f}점")
    col4.write("           ")
    
    col4.write("##### 눈을 감은 시간 측정을 통한 졸린 정도 측정")
    col4.write(f"Long time : 졸린 정도 : {(((left_eye_closed_cnt+right_eye_closed_cnt)/2)/video_info['total frame'] * 100):.2f}점")
  
    ### Total Score
    w1 = 0.4 #머리흔들기 가중치
    w2 = 0.3 #짝눈 가충치
    w3 = 0.1 #눈감기 가중치
    w4 = 0.2 #표정변화
    total_score = total_weight_score(hanging_score, unpair_blink_score, blink_score, emotion_score, w1, w2, w3, w4)
    col3.write("##### 행동 틱 위험도 점수화")
    col3.write(f"\n최종 틱 장애 위험도 : {total_score*100:.2f}점 : :N (0 - 30점 : 정상 | 30 - 60점 : 경증 | 60 - 100 : 중증) *w1 : {w1}, w2 : {w2}, w3 : {w3}, w4 : {w4}")
    
    
    return landmark,blink_frame,video_info,hanging_frame,res_image,results,emotion_x,emotion_y

### Check

### Check

#랜드마크 처리 
### landmakr[frame].multi_face_landmarks[person_cnt].landamrk[landmark number(1~478)]
# print(frame_len, person_len, landmark_len)

#########----------------------------------------------------------------###########
#landmark 2D image check    
def landmark_check(image, results, landmark_dict, video_info):
    temp = []
    for i in results.multi_face_landmarks[0].landmark:
        temp.append([i.x, i.y, i.z])
    temp_arr = np.array(temp)
    x_coords = temp_arr[:, 0]
    y_coords = temp_arr[:, 1]
    z_coords = temp_arr[:, 2]
    ## 변경 코드 : 정규화된 좌표를 픽셸 좌표로 변환
    x_coords *= video_info['width']
    y_coords *= video_info['height']
    
    l_eye_coords = temp_arr[landmark_dict['L_eye'], :]
    r_eye_coords = temp_arr[landmark_dict['R_eye'], :]
    mouth_coords = temp_arr[landmark_dict['Inner_lips'],:]
    face_coords = temp_arr[landmark_dict['Outline_face'], :]
    wn, ws, en, es = (np.min(face_coords[:, 0]), np.max(face_coords[:, 1])), (np.min(face_coords[:, 0]), np.min(face_coords[:, 1])), (np.max(face_coords[:, 0]), np.max(face_coords[:, 1])), (np.max(face_coords[:, 0]), np.min(face_coords[:, 1]))
    
    
    ## masked Landmark 2D ##
    ## 얼굴 영역에 빨간색 마스크 씌우기
    # 마스크 생성
    mask = np.zeros_like(image)
    mask[int(wn[1]):int(es[1]), int(wn[0]):int(en[0]), :] = [255, 0, 255]
    # 마스크와 원본 이미지 합치기
    cropped_image = cv2.bitwise_and(image, mask)
    result_image = cv2.addWeighted(image, 1, mask, 0.5, 0)

    col2_1.write("##### Masked 2D image with Landmark")
    fig1 = plt.figure(figsize=(16,8))
    plt.imshow(result_image)
    plt.axis('on')
    # plt.title('masked landmark 2D')
    col2_1.pyplot(fig1)

    # col2_1.write("##### Landmark 2D image")
    # fig1 = plt.figure()
    # plt.scatter(x_coords, y_coords, s=10, c= 'black', marker='o')
    # plt.plot(l_eye_coords[:,0], l_eye_coords[:,1], c= 'red', marker = 'o')
    # plt.plot(r_eye_coords[:,0], r_eye_coords[:,1], c= 'blue', marker = 'o')
    # plt.plot(mouth_coords[:,0], mouth_coords[:,1], c= 'orange', marker = 'o')
    # plt.plot(face_coords[:,0], face_coords[:,1], c= 'yellow', marker = 'o')
    # plt.xlabel('x axis')
    # plt.ylabel('y axis')
    # col2_1.pyplot(fig1)



    
    
    # 랜드마크이미지에서 박스 마스크만 가지고 와서 이를 image plot하기
    # 좌상, 좌하, 우상, 우하 좌표 가지고오기
    
    # plt.scatter(x_coords, y_coords, s = 10, c = 'black', marker = 'o')
    # #박스그리기
    # plt.gca().add_patch(Rectangle((wn[0], wn[1]), en[0]-wn[0], es[1]-wn[1], linewidth=2, edgecolor='red', facecolor='none'))
    # plt.show()
    # plt.close()
    
    # 박스 부분을 이미지에서 잘라오기
    # print(face_coords)
    # col2_2.write("##### Masked image with Landmark")
    # crop_image = image[int(wn[0]* video_info['height'])+5:int(en[0]* video_info['height'])+5, int(es[1]* video_info['width'])-7:int(wn[1]* video_info['width'])-7, :]
    # fig2 = plt.figure()
    # plt.imshow(crop_image)
    # plt.axis('off')
    # # plt.savefig('./result/analysis/mask_face.png')
    # col2_2.pyplot(fig2)
    
    ## masked Landmark 3D ##
    col2_2.write("##### Masked 3D image with Landmark")
    fig2 = plt.figure(figsize=(16,8))
    ax = fig2.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, c='black', marker='o')
    ax.plot(l_eye_coords[:,0], l_eye_coords[:,1], l_eye_coords[:,2], c='red', marker='o')
    ax.plot(r_eye_coords[:,0], r_eye_coords[:,1], r_eye_coords[:,2], c='blue', marker='o')
    ax.plot(mouth_coords[:,0], mouth_coords[:,1], mouth_coords[:,2], c='orange', marker='o')
    ax.plot(face_coords[:,0], face_coords[:,1], face_coords[:,2], c='yellow', marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev = 50, azim = 125)
    ax.axis('on')
    # plt.title('masked landmark 3D')
    col2_2.pyplot(fig2)





def landmarkarr(landmark):
    
    frame_len = len(landmark)
    # st.write('frame len:',frame_len)
    person_len = len(landmark[0]) #사람 몇명인지
    # st.write('person len:',person_len)
    landmark_len = len(landmark[0][0].landmark)
    # st.write('landmark len',landmark_len)
    landmark_arr = np.zeros((frame_len, person_len, landmark_len, 3))
    # Landmark Data preprocessing
    for frame_idx in range(frame_len):
        try:
            person_len = len(landmark[frame_idx]) #사람 몇명인지
        except:
            continue
        for person_idx in range(person_len):
            try:
                landmark_len = len(landmark[frame_idx][person_idx].landmark)
                for landmark_idx in range(landmark_len):
                    x,y,z, _= map(str, str(landmark[frame_idx][person_idx].landmark[landmark_idx]).split('\n'))
                    
                    landmark_arr[frame_idx, person_idx, landmark_idx] = x.split(' ')[-1],y.split(' ')[-1],z.split(' ')[-1] 
                    # print(landmark_arr)
            except:
                continue
    return landmark_arr
# print(landmark[0])
def landmark_analysis_plot(landmark_arr):

     # st.write(f"landamark_arr shape : {landmark_arr.shape}")
    frame_idx, person_idx = 0, 0 #frame 별, 사람 별
    landmarks_person_frame = landmark_arr[frame_idx, person_idx] #frame별 x,y,z 좌표 변화
    num_landmarks = landmarks_person_frame.shape[0]

    col2_3.write('###### 랜드마크 수의 변화를 나타내는 그래프')
    fig1 = plt.figure()
    plt.plot(range(num_landmarks), landmarks_person_frame[:, 0], label='X')
    plt.plot(range(num_landmarks), landmarks_person_frame[:, 1], label='Y')
    plt.plot(range(num_landmarks), landmarks_person_frame[:, 2], label='Z')
    plt.title('Landmark Coordinates over Landmark Index')
    plt.xlabel('Landmark Index')
    plt.ylabel('Coordinates')
    plt.legend()
    col2_3.pyplot(fig1)

    col2_4.write('###### x, y, z변화를 나타내는 그래프')
    fig2= plt.figure()
    for i in range(3): #landmark index별 x,y,z의 
        plt.plot(range(num_landmarks), landmarks_person_frame[:, i], label=f'Coordiante {i+1}')
    plt.title('X, Y, Z Coordiantes over Landmark Index')
    plt.xlabel('Landmark Index')
    plt.ylabel('Coordiantes')
    plt.legend()
    plt.tight_layout()
    col2_4.pyplot(fig2)
    
    col2_5.write('###### 전체 프레임 별 랜드마크에 대한 x,y,z 변화도') 
    landmark_idx= 219 #220번쨰 랜드마크 

    fig3= plt.figure()
    for person_idx in range(landmark_arr.shape[1]):
        plt.subplot(1, landmark_arr.shape[1], person_idx+1)
        x_coordiantes = landmark_arr[:, person_idx, landmark_idx, 0]
        y_coordiantes = landmark_arr[:, person_idx, landmark_idx, 1]
        z_coordiantes = landmark_arr[:, person_idx, landmark_idx, 2]
        plt.plot(range(landmark_arr.shape[0]), x_coordiantes, label=f'X')
        plt.plot(range(landmark_arr.shape[0]), y_coordiantes, label=f'Y')
        plt.plot(range(landmark_arr.shape[0]), z_coordiantes, label=f'Z')
        plt.title(f'All frame for Landmark {landmark_idx+1} at Person : {person_idx+1}')
        plt.xlabel('Frame')
        plt.ylabel(f'Coordiantes')
        plt.legend()
    plt.tight_layout()
    col2_5.pyplot(fig3)
    
def scatter_3d(landmark_arr):
    st.subheader("3D 산점도를 이용한 랜드마크의 위치별 인덱스 확인")
    reshaped_arr = landmark_arr[0,0].reshape(-1,3)
    landmark_df = pd.DataFrame(reshaped_arr, columns=['X', 'Y', 'Z'])
    # landmark_df
    # 3D 산점도 그리기
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(landmark_df['X'], landmark_df['Y'], landmark_df['Z'], s=10, )  # s는 점의 크기를 나타냅니다.

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot of Landmarks')

    st.pyplot(fig)


def area_analysis(landmark_arr, tag):
    fig = plt.figure(figsize=(int(f"{landmark_arr.shape[1]*6}"),6))
    # 사람별로 프레임별 면적 계산
    for person_idx in range(landmark_arr.shape[1]):
        plt.subplot(1, landmark_arr.shape[1], person_idx+1)
        plt.plot(range(landmark_arr.shape[0]), landmark_arr[:,person_idx], label=f'{person_idx+1} person')
        plt.title(f'Compare All frame for {tag}')
        plt.xlabel('Frame')
        plt.ylabel(f'Area')
        plt.legend()
    col2.pyplot(fig)
    
# 삼각형 메쉬 데이터 면적 계산
def calculate_triangle_area(vertices):
    # 세점 A,B,C
    A, B, C = vertices
    
    #벡터 AB, AC 계산
    AB = B - A
    AC = C - A 
    
    #외적 계산
    cross_product = np.cross(AB, AC) 
    
    # 삼각형 넓이 계산
    area = 0.5 * np.linalg.norm(cross_product)
    
    return area
#부위의 모든 랜드마크 면적 계산 누적 함수 
def cal_area(landmark, landmark_dict, tag):
    # 각 점 x,y 좌표 추출
    try:
        iris_landmark = landmark[:, :, landmark_dict[tag], :]        
        # print(f' iris_landmark shape : {iris_landmark.shape}')
        # 각 프레임별 사람별 면적 저장하기 
        area_fr_per = np.zeros((iris_landmark.shape[0], iris_landmark.shape[1]))
        # 각프레임별 루프
        for frame_idx in range(iris_landmark.shape[0]):
            #각 사람별 루프
            for person_idx in range(iris_landmark.shape[1]):
                # 각 랜드마크별로 루프
                mesh_vetrices = iris_landmark[frame_idx, person_idx, :, : ]
                total_area = 0. 
                for landmark_idx in range(1, iris_landmark.shape[2]-1):
                    vetrices = np.array([
                        iris_landmark[frame_idx, person_idx, 0, :],
                        iris_landmark[frame_idx, person_idx, landmark_idx, :],
                        iris_landmark[frame_idx, person_idx, landmark_idx+1, :],
                        ])
                    total_area += calculate_triangle_area(vetrices)
                area_fr_per[frame_idx, person_idx] = total_area
                      
    except:
        print('Error')

    # print(f"{tag} area shape : {area_fr_per.shape}")

    return area_fr_per

def area_dict_func(landmark_arr):
    landmark_dict = {
        #왼쪽눈
        'L_eye' : [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7], #시계방향 9시부터 시작
        #오른쪽눈
        'R_eye' : [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382], #시계방향 9시부터 시작
        #왼쪽 눈동자
        'L_iris' : [470, 471,469, 472, 468], #위에서부터 Up, Left, Right, Down, Center position
        #오른쪽 눈동자
        'R_iris' : [475, 476, 474, 477, 473],
        #왼쪽 눈썹
        'L_eyebrow' : [70, 63, 105, 66, 107, 55, 65, 52 ,53, 46], #시계방향 9시부터 시작
        #오른쪽 눈섭
        'R_eyebrow' : [336, 296, 334, 293, 300, 276, 283, 282, 295 ,285], #시계방향 9시부터 시작
        #윗 입술
        'Up_lips' : [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 391, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],#시계방향 9시부터 시작
        #아랫 입술
        'Down_lips' : [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 391, 375, 321, 405, 314, 17, 84, 181, 91, 146],#시계방향 9시부터 시작
        #입술 안
        'Inner_lips' : [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],#시계방향 9시부터 시작
        #얼굴외곽
        'Outline_face' : 
            [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284,
            251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93],
        'L_eye_blink' : [33, 159, 133, 145],
        'R_eye_blink' : [362, 386, 263, 374], 
    }
    L_eye_area = cal_area(landmark_arr, landmark_dict, 'L_eye')
    R_eye_area = cal_area(landmark_arr, landmark_dict, 'R_eye')
    L_iris_area = cal_area(landmark_arr, landmark_dict, 'L_iris')
    R_iris_area = cal_area(landmark_arr, landmark_dict, 'R_iris')
    L_eyebrow_area = cal_area(landmark_arr, landmark_dict, 'L_eyebrow')
    R_eyebrow_area = cal_area(landmark_arr, landmark_dict, 'R_eyebrow')
    Up_lips_area = cal_area(landmark_arr, landmark_dict, 'Up_lips')
    Down_lips_area = cal_area(landmark_arr, landmark_dict, 'Down_lips')
    Inner_lips_area = cal_area(landmark_arr, landmark_dict, 'Inner_lips')

    area_dict = {
        'L_eye_area' : L_eye_area,
        'R_eye_area' : R_eye_area,
        'L_iris_area' : L_iris_area,
        'R_iris_area' : R_iris_area,
        'L_eyebrow_area' : L_eyebrow_area,
        'R_eyebrow_area' : R_eyebrow_area,
        'Up_lips_area' : Up_lips_area,
        'Down_lips_area' : Down_lips_area,
        'Inner_lips_area' : Inner_lips_area
    }
    return area_dict

#짝눈 Treshold 정하기 위한 Wavelet Transformed data
def eyes_pair_to_wavelet(area_dict):
    eyes_pair_area_ratio = np.array([])
    for i in range(len(area_dict['L_eye_area'])):
        eyes_pair_area_ratio = np.abs(area_dict['L_eye_area'] - area_dict['R_eye_area'])
    signal = eyes_pair_area_ratio[:,0]
    wavelet = wavedec(signal, 'db1', level =1, mode = 'symmetric')[0]
    padding_wavelet_ratio = np.repeat(wavelet, 2)
    #복원된 신호와 원본 신호 시각화

    col3_3.write("#### Original Unpair Eye Blink Ratio to Wavelet Transformed Ratio")
    fig1 = plt.figure()
    plt.plot(signal, label='Original Unpair Eye Blink Ratio', color='blue')
    plt.plot(padding_wavelet_ratio, label='Wavelet Transformed Ratio', color='red', linestyle='--')
    plt.title('Original and Reconstructed Signal')
    plt.legend()
    # plt.savefig('./result/analysis/eye_blink_wavelet.png')
    col3_3.pyplot(fig1)

def blink_ratio(blink_frame):
    blink_arr = np.array(blink_frame)
    # left_norm, right_norm = blink_z_score_norm(blink_arr[:,1:])
    left_norm, right_norm = blink_arr[:,1], blink_arr[:,2]
    
    fig1 = plt.figure()
    plt.plot(range(len(left_norm)), left_norm)
    plt.title('Left Blink Aspect Ratio')
    
    fig2 = plt.figure()
    plt.plot(range(len(left_norm)), right_norm)
    plt.title('Right Blink Aspect Ratio')
    # plt.legend()
    col3_1.write("#### Left Eye Blink Aspect Ratio")
    col3_1.pyplot(fig1)
    col3_2.write("#### Right Eye Blink Aspect Ratio")
    col3_2.pyplot(fig2)

# frame 별 hanging seq 분석
def hanging_seq(video_info,hanging_frame):
    col4_1.write("#### Shaking Head Sequence per Frame")
    hanging_array = np.zeros((video_info['total frame'],1))
    

    # hanging_array[np.array(hanging_frame[:,0]).astype(int)] = np.array(hanging_frame)[:,1]
    for idx, point in enumerate(np.array(hanging_frame)[:,0].astype(int)):
        if point < len(hanging_array):
            hanging_array[point] = hanging_frame[idx][1]*10

    # hanging_array를 플로팅
    fig = plt.figure()
    plt.plot(range(1, video_info['total frame']+1), hanging_array, label='Hanging Sequence')
    plt.xlabel('Frame Index')
    plt.ylabel('Hanging Value')
    plt.title('Hanging Sequence Over Frames')
    plt.legend()
    col4_1.pyplot(fig)

    # """
    #     hanging threshold 정하기 -> 0.05 이상이면 hardly hanging 으로 판단
    # """


######################################################################################################################################
######################################################################################################################################
   
tab1, tab2,tab3,tab4,tab5= st.tabs(['Video and Score' , 'Landmark Analysis','Eye Blink Analysis','Shaking head Analysis','Emotion Analysis'])   
# uploaded_file = st.sidebar.file_uploader("동영상 파일을 업로드하세요.", type=["mp4"])

landmark_dict = {
        #왼쪽눈f
        'L_eye' : [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7], #시계방향 9시부터 시작
        #오른쪽눈
        'R_eye' : [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382], #시계방향 9시부터 시작
        #왼쪽 눈동자
        'L_iris' : [470, 471,469, 472, 468], #위에서부터 Up, Left, Right, Down, Center position
        #오른쪽 눈동자
        'R_iris' : [475, 476, 474, 477, 473],
        #왼쪽 눈썹
        'L_eyebrow' : [70, 63, 105, 66, 107, 55, 65, 52 ,53, 46], #시계방향 9시부터 시작
        #오른쪽 눈섭
        'R_eyebrow' : [336, 296, 334, 293, 300, 276, 283, 282, 295 ,285], #시계방향 9시부터 시작
        #윗 입술
        'Up_lips' : [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 391, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],#시계방향 9시부터 시작
        #아랫 입술
        'Down_lips' : [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 391, 375, 321, 405, 314, 17, 84, 181, 91, 146],#시계방향 9시부터 시작
        #입술 안
        'Inner_lips' : [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],#시계방향 9시부터 시작
        #얼굴외곽
        'Outline_face' : 
            [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284,
            251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93],
        'L_eye_blink' : [33, 159, 133, 145],
        'R_eye_blink' : [362, 386, 263, 374], 
    }

with tab1:
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    uploaded_file = st.sidebar.file_uploader("동영상을 업로드 해주세요.", type=["mp4"])

    if uploaded_file is not None:
        # 업로드된 동영상을 임시 파일에 저장
        
         landmark,blink_frame,video_info,hanging_frame,res_image,results,emotion_x,emotion_y = upload_video(upload=uploaded_file)
        # landmark = make_landmark(upload = uploaded_file)
        # ## 랜드마크 프레임별 저장 함수 실행
        # landmark_arr = landmarkarr(landmark)
        # st.write(landmark_arr)
with tab2:
    col2_1, col2_2 = st.columns(2)
    col2_a,empty2_a =st.columns([0.9,0.1])
    col2_3, col2_4, col2_5 = st.columns(3)
    col2_b,empty2_b = st.columns([0.9,0.1])
    if uploaded_file is not None:
        
        landmark_arr = landmarkarr(landmark)
        landmark_check(res_image, results, landmark_dict, video_info)
        col2_a.write("<1) 틱 장애 의심 환자의 얼굴의 각 랜드마크들의 좌표를 2D이미지로 시각화 이미지>")
        col2_a.write("<2) 첫 번째에서 추출한 랜드마크 좌표들을 틱 장애 의심 환자의 얼굴에 masking한 이미지>")
        col2_a.write("          ")
        col2_a.write("          ")
        col2_a.write("          ")
        landmark_analysis_plot(landmark_arr)
        col2_b.write("<1)랜드마크 index에 대한 랜드마크 수의 변화를 나타낸 그래프>")
        col2_b.write("<2)랜드마크의 index별 x,y,z 성분들의 변화를 나타내는 그래프>")
        col2_b.write("<3)프레임별 랜드마크에 대한 x,y,z축 좌표에 대한 변화도 그래프> :N 각 프레임에서 변화도를 이용하여 머리가 흔들린 방향과 정도를 시각화")
        

with tab3:
    col3_3, empty = st.columns([0.7,0.3])
    col3_a,empty =st.columns([0.9,0.1])
    col3_1, col3_2 = st.columns(2)
    col3_b,empty = st.columns([0.9,0.1])
    if uploaded_file is not None:
        
        area_dict = area_dict_func(landmark_arr)
        eyes_pair_to_wavelet(area_dict)
        col3_a.write("<Unpair-eye Closed, 윙크하듯이 한 쪽 눈만 감는 부분을 감는 부분에 대한 Treshold를 감지하기 위한 Wavelet Transform 그래프>")
        col3_a.write("*Wavelet Transform을 통해서 그래프 함수의 변화율을 유연하게 만들고 너무 빠른 감지하여 변하는 부분을 보완할 수 있다. ")
        col3_a.write(" 짝눈(Unpair Eye) 감는 횟수에 대한 분석 중 threshold를 0.0004( > : Non-tic)정도로 유지하는 것이 좋음을 확인 할 수 있다.")
        col3_a.write("       ")
        col3_a.write("       ")
        col3_a.write("       ")
        
        blink_ratio(blink_frame)
        col3_b.write("<1) 왼쪽 눈의 가로, 세로 종횡비 변화를 나타낸 그래프>")
        col3_b.write("<2) 그래프는 오른쪽 눈의 가로, 세로 종횡비를 나타낸 그래프>")
        col3_b.write("프레임별 눈의 크기에 대한 양상비를 종횡비 값을 이용하여 수치가 변화가 큰 경우 Blink로 감지하는 알고리즘")


with tab4:
    col4_1, empty = st.columns([0.7,0.3])
    col4_a,empty = st.columns([0.9,0.1])
    if uploaded_file is not None:
        
        hanging_seq(video_info,hanging_frame)
        col4_a.write("<해당 그래프는 프레임별 머리가 흔들리는 틱이 있었는지를 나타내는 그래프>")
        col4_a.write("일반 사람들도 수업 시간에 일반적으로 고개를 들고 움직이는 Treshold 값을 0.5이하로 설정하였음.")
        col4_a.write("그러므로 해당 그래프의 0.5이상인 값들을 통해 영상에서 틱 장애 의심 환자가 영상의 재생 시간동안 틱이라고 생각될 수 있는 머리 흔들림 증상의 횟수를 한 눈에 관찰할 수 있다.")

with tab5:
    col5_1, empty = st.columns([0.7,0.3])
    col5_a,empty = st.columns([0.9,0.1])
    if uploaded_file is not None:
        
        emotion_bar(emotion_x,emotion_y)
        col5_a.write("<틱 장애 의심 환자의 표정을 통한 감정 예측을 나타내는 그래프>")
        col5_a.write("일반 사람들은 보통 수업 시간에 무표정을 하고 있는 경우가 대부분이다. :N 하지만 틱 장애 환자들은 갑작스러운 표정 변화와 같은 얼굴 찡그림 같은 증상을 나타낸다.")
        col5_a.write("해당 분석을 통해서 틱 장애 의심 환자의 표정 변화를 관측하고 이를 틱 장애 감지 보조 지표로 사용해보았다.")
        col5_a.write("Tic : 4가지 이상의 감정변화가 존재, Non-Tic :3가지 이하의 감정변화가 존재")
