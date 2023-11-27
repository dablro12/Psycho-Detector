#두개의 constant value를 정의
EYE_AR_THRESH = 0.23 #baseline code thresholld
EYE_AR_CONSEC_FRAMES = 3

# eye landmark model 파일 
eye_landmarks ="model/shape_predictor_68_face_landmarks.dat"

# 초기설정
# initalize the frame counters and the total number of blink 
COUNTER = 0 
TOTAL = 0 