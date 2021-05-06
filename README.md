## Face Mask Detector

사람의 얼굴을 인식하여 마스크의 착용 유무를 판별해주는 모델입니다.

---

코로나 바이러스로 인해 마스크 착용이 의무화되면서 음식점과 카페 등 여러 사람이 이용하는 다중시설에 가면 발열체크와 마스크 인식 기계를 흔히 볼 수 있습니다. 평소에는 그냥 지나쳤지만 컴퓨터 비전을 공부하다보니 관심이 생겨서 비슷하게 만들어보았습니다.


### 환경
Python으로 구동가능합니다.
모델은 **tensorflow**로 생성되었으며 얼굴 인식에는 **cvlib** 오픈소스 라이브러리를 사용했습니다. 

#### 설치 목록
- Python 3.6.9
- tensorflow-gpu 2.3.0
- cvlib 0.2.6
- opencv-python 4.4.0.42
- numpy 1.19.2

### 구동
```bash
├─data
│  ├─dataset
│  │  ├─mask
│  │  └─nomask
│  ├─mask
│  ├─nomask
│  └─sample
│      ├─dataset
│      │  ├─mask
│      │  └─nomask
│      ├─mask
│      ├─nomask
│      └─test
└─models
```
#### 직접 모델 만들어 보기
1. `/data/mask`와 `/data/nomask`에 이미지를 수집한다.([블로그 글](https://jungwon-moon.github.io/face%20mask%20detector/make_dataset/) 참고)
2. `make_dataset.py`를 실행한다.
3. `/data/dataset/mask`와 `/data/dataset/nomask`에 정제된 이미지 파일을 확인한다.
4. `train_model.py`를 실행한다.


#### 모델 실행하기
- 이미지
> python run_fmdm_image.py "이미지 파일 경로"
*ex)python run_fmdm_image.py "C:/image.png"*

- 영상
> python run_fmdm_video.py "비디오 파일 경로"
*ex)python run_fmdm_video.py ""*

- 캠 *(아쉽게도 캠이 없어서 만들지 못함)*

### 참고
- tensorflow 공식문서
- https://bskyvision.com/1082