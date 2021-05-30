# 인공지능 프로젝트

김정천,김준홍,김영래
### 프로젝트 개요
저희가 Numpy, Cupy로 구현한 CNN모델로 COVID CT 사진을 보고 분류하는 작업을 해보려 하였습니다.

하지만, 저희가 구현한 모델의 학습이 원활히 되지 않아 이유를 찾기 위해 lr 조절, optimizer 문제, 모델 구조 문제 등 여러가지 이유를 찾아보았지만 원인을 찾지 못하여 케라스 라이브러리를 사용해서 COIVD CT 사진을 분류하는 모델을 여러가지 구현하여 비교해보았습니다.

(https://www.kaggle.com/maedemaftouni/large-covid19-ct-slice-dataset)

### How to do
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

### 결과

VGG16

![image](https://user-images.githubusercontent.com/30318926/120109400-3f8dff80-c1a4-11eb-9edd-a724fe89a330.png)

train acc 그래프가 매우 이상하게 나왔습니다.

loss값도 1밑으로 잘내려가지 않는 모습을 보였습니다.

train acc 그래프의 모양이 저희가 구현에 실패했던 acc와 비슷한 모양을 보이는 것을 보아 어쩌면 이러한 단순 CNN 모델의 고질적 문제인 vanishing gradient의 문제일수도 있다 생각이 들었습니다.

layer가 deep하다고 항상 성능이 좋게 나오는 것이 아니라는 것을 보여주는 좋은 예 인 것 같습니다. parameter 수 또한 압도적으로 높으며 그에 따라 학습 시간도 가장 긴 모습을 보여줍니다.

10 epochs train acc: 0.44087985157966614

test acc : 0.44152048230171204

LENET

![image](https://user-images.githubusercontent.com/30318926/120109502-a7dce100-c1a4-11eb-941c-cd14ef857e08.png)

빠르고 안정적으로 높은 train acc를 보여줬습니다. 

parameter 수도 적고 layer가 얕음에도 불구하고 성능이 좋고 속도도 매우 빠른 것을 볼 수 있습니다.

10 epochs train acc: 0.9576877951622009

test acc: 0.955962598323822

RESNET50

![image](https://user-images.githubusercontent.com/30318926/120109508-ae6b5880-c1a4-11eb-9435-e5e93e97f746.png)

train acc가 급하게 올라가진 않았지만 서서히 안정적으로 높은 train acc에 진입하는 모습을 보여줍니다.

layer는 가장 깊지만 parameter 수는 그렇게 많지 않았고 epoch 10 까지만 했을 때 그래프의 모양이 상향 그래프여서 15epochs까지 학습시켰습니다.

vanishing gradient를 해결하고 layer 마저 깊게 쌓아 성능이 가장 뛰어난 것을 볼 수 있습니다. 하지만 속도는 LENET에 비해서는 많이 느린편입니다.

15 epochs train acc: 0.9758842587471008

