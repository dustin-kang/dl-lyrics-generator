# Proj4_DLProject
딥러닝 프로젝트

# 프로젝트 설명
### 📊 1. 관심 데이터 선정
"지금 무엇을 해야 할까?"는 연구자에겐 가장 어려운 질문 중 하나입니다. 그렇기에 석사과정 혹은 사원급에 해당하는 초기 연구자는 연구소(혹은 회사)에서 하던 연구방향에서 +@를 구현하는 것이 대부분입니다.
하지만 우리는 아직 연구소(혹은 회사)가 없는 상태입니다. 이런 상황에서 연구 주제를 정해야 한다면 주제는 떠오르지 않고, 써보고 싶은 Method만 떠오르는 것이 일반적입니다.
그렇기 때문에 이번 프로젝트에서는 내가 취직하고 싶은 회사와 그 회사의 연구/사업 내용을 알아보고 해당 회사에서 풀고자 하는 문제와 비슷한 데이터를 찾아봅니다.
그리고 그에 맞는 기술력을 키워보는 것을 목표로 하는 것이 좋습니다.

> - 현대자동차, 소카 등의 자율주행차 관련 업무를 목표로 하고 있다면, 관련 기술을 시험해볼만한 데이터셋이 있는 지 찾아보아야 한다.
> - Kakao i(카카오) 나 Clova(네이버)와 같은 서비스를 목표로 한다면 음성 인식, 음성 합성에 사용할 수 있는 데이터 셋을 찾아볼 수 있습니다.
> - 네이버 웹툰 등을 목표로 한다면, GAN등의 생성모델과 보안(Fingerprint recognition) 관련 딥러닝 기술을 적용해 볼 수 있는 데이터 셋을 찾는 게 좋습니다.
> - Airbnb와 같은 대고객 서비스를 만들고 싶다면 고객 데이터의 통계분석이 필요하며, 추천시스템을 적용해 볼 수 있는 데이터를 찾는 것이 좋습니다.

### ❓ 2. 데이터 선정 이유
여러분이 선정한 데이터와 그 데이터를 가공하면서 얻은 지식과 경험을 "어떤 회사에서 높이 살 수 있을까?", "어디 회사의 어느부분에 적용해 볼 수 있을까"를 생각해서 기록하여 봅니다.


### 🧩 3. 데이터를 이용한 가설 수립
데이터를 선정함과 동시에 데이터를 통해서 내가 무엇을 해볼 수 있을 지 가설을 세우는 것이 중요합니다.
첫 번째로는 쓸모있는 가설, 즉 이유가 명확한 가설이어야 합니다.
데이터기반의 사고방식(Data-driven Thinking)에 대한 마인드셋을 Section 1과 2에서 배웠습니다.

이번 프로젝트에서는 이를 심화시켜서 진짜 필요한 기술을 찾아봅시다.
적어도 내 생각에는 정말 쓸모있다고 생각할 수 있는 스토리라인을 만들어보세요!


### 🧹 4. 데이터 전처리
가설을 정했다면, 데이터의 가공을 시작해봅니다.
바로 모델에 적용할 수 있는 데이터도 있겠지만 데이터를 직접 보며 전처리를 해보는 것을 추천드립니다.

> #### 데이터 전처리 예시
> - 데이터의 정규화(Normalization)
> - 노이즈 및 이상치(Outlier) 제거
> - 타겟 레이블(Label or Ground Truth) 생성 혹은 선택 등

### 🧠 5. 딥러닝 방식 적용
적용에 앞서 내가 가진 문제를 굳이 딥러닝을 적용해야 하는 지 확인할 필요가 있습니다.
신경망 첫 시간에 엄청 큰 검을 들고 스테이크를 썰던 이미지를 기억하시나요?

딥러닝의 가장 큰 장점은 "어려운 문제를 더 어렵게 풀지만, 그 결과가 끝내주게 좋다"는 것인데요.
만약 너무 쉬운 문제에 딥러닝을 적용한다거나 DL이 아닌 ML 방법론들보다 더 많은 리소스를 하는데도 성능이 낮으면 안되겠죠?


### ✔️ 6. Chance Level 이 넘는지 확인
이진 분류(Binary Classification)는 Chance level이 0.5(50%)인 문제입니다. 말 그대로 하나로 찍는 머신이 이라도 약 50%정도는 달성한다는 뜻인데요. (데이터가 편향된 경우는 제외)
여러분이 선택한 문제가 MNIST와 같은 10개 클래스를 가진 다중 분류라면 Chance level이 0.1(10%)이 되겠죠.

여러분이 선택한 문제에 딥러닝을 적용한 결과가 Chance level 보다 월등하게 좋은 성능을 기록하는 지를 체크해봅니다.
그렇지 않다면 데이터를 다시 들여다 보거나 모델을 다시 뜯어보면서 내 가설이 틀렸을 수 있다는 것을 확인하여 봅니다.


### 🔍 7. 모델 검증(Validation)
모델을 만들어서 어느정도 성능이 나왔다면, CV을 통해서 일반화될 가능성이 있는 지 확인해봅니다.
K-Fold 교차 검증을 통해 일반화가 어느정도 되는지 알 수 있습니다.
더불어 하이퍼파라미터를 변경하면서 최적화까지 해 볼 수 있겠습니다.
물론 해당 과정에서 너무 많은 시간이 소요되지 않도록 잘 조정해야 합니다.


### 📋 8. (Option) Requirements.txt 제작 및 재구현
언제까지 Colab만 쓰지는 않을 것입니다.
여러분이 만든 딥러닝 모델을 다시 사용할 수 있도록 저장하고 새로운 환경에서도 바로 동작할 수 있도록 Requirements.txt를 만들어봅니다.
만든 Requirements.txt 를 이용하여 가상환경 혹은 독립된 PC에서 같은 프로젝트를 진행하여 봅니다.




# 🖍 프로젝트 채점 기준
아래 4가지 질문 사항을 모두 만족하여야 합니다.

### "여러분이 풀고자 하는 문제를 잘 설정하였는가?"
여러분이 선택한 데이터셋을 사용하여 어떤 문제를 풀고자 하는 지를 영상에서 잘 설명해주세요.
단순히 익숙하다는 이유로 MNIST, cifar100 데이터를 사용하면 안되겠죠?
여러분만의 문제와 데이터셋을 마련하는 것이 이번 프로젝트의 첫 번째 목표입니다!

### "문제를 풀기 위한 모델 선택을 알맞게 설정하였는가?"
여러분이 선택한 모델에 대한 구조를 설명할 필요는 없으며 해당 모델을 왜 선택하였는 지에 대해서 설명해주세요.
예를 들어, 이미지 분류 문제에 단순한 LSTM을 적용하면 안 될 겁니다.
풀고자 하는 문제와 알맞는 모델을 잘 선택하는 것이 이번 프로젝트의 두 번째 목표입니다!

### "모델 학습을 제대로 진행하였는가?"
데이터를 어떻게 전처리하였고 모델에 입력하였는 지를 설명해주세요.
모델의 성능이 좋아야 할 필요는 없겠지만 모델 학습은 진행되어야 합니다.

### 완성하지 못한 부분에 대하여 "한계점과 추후 해결 방안을 알맞게 작성하였는가?"
프로젝트 기간 동안 제시한 문제에 대해서 만족할 만한 성능을 얻지 못할 수도 있습니다.
여러분이 만족할 만한 결과를 위해서 어떤 방향으로 프로젝트를 발전시켜 나가야 할지 구체적으로 설명해주세요.

영상 길이는 5분 이상 10분 이내로 작성합니다. 너무 길어지거나 너무 짧아지지 않도록 해주세요!
