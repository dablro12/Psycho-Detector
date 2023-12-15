# Psycho-Detector 🧠
# Short-term Muscular Tic-Patients Detector : Automated Tic Detector Algorithm based model using Google MediaPipe, CNN Classifier to Analysis Tic Region of Interest
#### Project nickname : Psycho Detector
#### Member : 최대현(PM, Engineer), 정영진(Engineer), 이상경(Data Analysist), 이수하(Data Analysist) 
#### Project execution period : 2023.09~2023.12
#### Project Hosting : [최대현](https://www.notion.so/Medical-Image-Processing-Psycho-Detector-AI-b535ea49d0e74ac9ac5a7dfee8f3df6b?pvs=4)
-----------------------

## 0. Service description
### Orginal Video(Input) / Tic Detector Video(Output)
![original test_file](https://github.com/dablro12/Psycho-Detector/assets/54443308/7487dfe2-5301-4347-884d-f6484def0e88) ![ezgif com-video-to-gif](https://github.com/dablro12/Psycho-Detector/assets/54443308/6b64c9bb-1ff7-4158-877e-1ac6a7203812)



### Streamlit Page
![1](https://github.com/dablro12/Psycho-Detector/assets/54443308/3cf6ac35-2008-45fc-a9c1-54aa2101ec52)
![2](https://github.com/dablro12/Psycho-Detector/assets/54443308/ff70587b-2e94-415a-94d8-da0ab8b54c6b)

## Description & Purpose
졸음감지 시스템의 행동 패턴 인식 활용하여 1. 심리장애(뚜렛 증후군 및 ADHD 증후군) 감지 및 2. 수업 중 강의 집중력 향상 및 강의 경험 개선 방향을 제시한다.

- 학령기 아동을 대상으로 매우 흔하게 발생하는 ‘틱’ 패턴을 감지하여 추후에 발생할 수 있는 뚜렛증후군이나 ADHD증후군과 같은 심리 장애에 대하여 초기에 적절한 조치가 취해질 수 있도록 한다.
- 사용자가 집중력을 잃고 피로하거나 스트레스를 받는 경우, 휴식을 제안하는 메시지나 음악 선택 등의 집중력 향상 방안 및 학습 평가 결과를 제시해 사용자의 학습 개선 방안을 제공한다.
- 사용자가 무의식적으로 나타내는 강의 내용의 이해와 관련된 행동(고개 끄덕임, 얼굴 찡그림, 표정 등)을 파악하여 강의자의 지도 개선에 도움을 준다.

![스크린샷 2023-11-30 오후 4 58 22](https://github.com/dablro12/Psycho-Detector/assets/54443308/dbb51943-5bc1-4a7b-869b-e55ed2f63a60)
Pipeline

### 1. function list
|구분|기능|구현|
|------|---|---|
|S/W|얼굴 메쉬 데이터 생성 모델 |Mediapipe Face Landmark Detection with google|
|S/W|각도 측정 데이터 변환 및 분석 |OpenCV/Numpy/Pandas/Matplotlib|
|S/W|Visualization|Streamlit|
|H/W|입력 모듈|Iphone 12 pro|

### 2. detailed function
#### Software
**- 마스크 및 Featrue Abstract **
- face_outline_masking : 얼굴 외곽 마스크 생성 함수
- hanging_cnt_v1 : 머리 Hanging Detection 부호화 이용 함수 
- hanging_cnt_v2 : 머리 Hanging Detection 그룹화 이용 함수
- is_eye_closed : 종횡비를 이용한 눈 감기 횟수 측정 함수
- face_emotion_detector : 표정 감지 딥러닝 모델 inference 함수
- save_area : Mesh data 부피 저장 함수
- z_score_norm : Z_score Normalization 함수
  
**- Scoring : Feature Scoring **
- total_weight_score : 최종 스코어 함수
- hanging_face_score : 머리 Hanging Scoring 함수
- unpair_eye_blink_score : 눈 찡그림 감지 및 Scoring 함수
- blink_equal_score : 눈 감기 Scoring 함수
- emotion_scoring : 표정 감지 Scoring 함수
  
**- Analysis : 얼굴 내 존재하는 특징(눈, 얼굴 등 특정 부위의 영역과 좌표(x,y,z) **
- save_np : 프레임별 랜드마크 저장 함수
- landmark_analysis_plot : 랜드마크 분석 함수
- calculate_triangle_area : 메쉬 데이터 면적 계산 함수
- cal_area : 메쉬 데이터 모든 랜드마크 면적 계산 함수
- area_analysis : 프레임별 면적 분석 함수
- blink_z_score_norm : 눈 감기에 대한 적정 Threshold 선정 분석 함수
- box_plot_seperate : 테스트 셋 검증 함수  
- 시각화 : Streamlit를 이용하여 웹페이지로 구현가능한 웹서비스


## Environment

> Python Version 3.8.18
> pytorch latest version
> Linux Ubuntu


## Prerequisite

> impor cv2
>
> import mediapipe as mp
>
> import os 
>
> import numpy as np
>
> import pandas as pd
>
> import matplotlib.pyplot as plt
>
> import seaborn as sns
>
> from mpl_toolkits.mplot3d import Axes3D
>
> from mediapipe import solutions
>
> from mediapipe.framework.formats import landmark_pb2
>
> from pywt import wavedec
>
> from imutils.video import VideoStream
>
> import f_detector
>
> import imutils
> 

## Files
`f_detector.py` Face Detector code file

`config.py` f_detector.py config file

`main.py` Main code file by using python language 

`main_for_human.ipynb` human detect updating main code file

`streamlit.py` Web application to visualization our project 

## Usage Pipeline
`main.ipynb`

## Usage Solution
1) Download `test`, `model` folder and  `streamlit.py` file
2) Execute terminal on your download path, `streamlit run streamlit.py`
