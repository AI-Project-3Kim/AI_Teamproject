# 인공지능 프로젝트

김정천,김준홍,김영래

전처리를 이미 한 상태의 데이터를 다운 받을 수 있게 하여 전처리 코드는 안돌려도 됩니다.

- cupy로 구현한 모델 실행하기

1. cupy를 사용할 수 있게 CUDA Tookit를 설치하고 버전에 맞는 Cupy를 pip을 사용하여 설치합니다.

2. https://github.com/AI-Project-3Kim/AI_Teamproject.git로 git clone을 받아옵니다.

3. https://drive.google.com/file/d/1lteQ-ZbJx1AFrXyqgIUuyGziS2Vpfbml/view?usp=sharing 에서 x.npy파일을 다운 받고

4. https://drive.google.com/file/d/1sF8PvAQg2AktrXuXM5KLq4fwJn2tIBIr/view?usp=sharing 에서 y.npy파일을 다운받아서

5. 프로젝트 최상위 폴더의 Dataset폴더에 복사를 합니다.

6. 최상위 폴더에서 " python covid.py " 을 실행 시키면 됩니다.


- keras로 구현한 모델 실행하기 (1,2번 위와 동일)

3. https://drive.google.com/file/d/1BqQmzkw4IvX31QZ93biRuMrwNdPxlb3p/view (사진 data)를 받습니다.

4. 받은 사진 data를 프로젝트 최상위 폴더에 압축을 풉니다.

5. LENET.py와 RESNET50.py, VGG16.py 3개의 python 파일을 python으로 실행시키면 됩니다.
