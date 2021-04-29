# 이미지에서 사람의 얼굴을 찾은 후 dataset 폴더에 저장한다.

import os
import glob
import cv2
import cvlib as cv

# 파일 구조 확인
if not os.path.isdir('/data'):
    os.mkdir('/data')

os.chdir('./data')
path = os.getcwd()

if not os.path.isdir(path+'/dataset'):
    os.mkdir(path+'/dataset')
if not os.path.isdir(path+'/dataset/mask'):
    os.mkdir(path+'/dataset/mask')
if not os.path.isdir(path+'/dataset/nomask'):
    os.mkdir(path+'/dataset/nomask')


# 파일 경로 변수로 저장
mask_files = glob.glob(path+'/mask/./*')
nomask_files = glob.glob(path+'/nomask/./*')

def make_dataset(files, label="nomask"):
    count = 0   # 파일이름으로 사용
    
    for file in files:
        try:
            img = cv2.imread(file)
            faces, conf = cv.detect_face(img, threshold=0.6)

            if faces == []:
                continue
            
            (h, w) = img.shape[:2]
            for face in faces:
                
                if face[2] > w or face[3] > h:  # bbox가 잘못 지정된 경우 제외
                    continue
                count += 1
                face_img = img[face[1]:face[3],face[0]:face[2]]
                cv2.imwrite(f"dataset/{label}/{count:04}.jpg", face_img)
        except:
            print(file)     # 에러 파일 확인


make_dataset(mask_files, "mask")        # mask   레이블 생성
make_dataset(nomask_files, "nomask")    # nomask 레이블 생성