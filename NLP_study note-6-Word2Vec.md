# 딥 러닝을 이용한 자연어 처리 입문     

​      

## Word2Vec     

단어 간 유사도를 반영할 수 있도록 단어의 의미를 벡터화 할 수 있는 방법이 필요!     

이를 위해서 사용되는 대표적인 방법     

<br/>

## 분산 표현 (Distributed Representation)     

희소 표현 (Sparse Represenatation)은 각 단어간 유사성을 표현할 수 없다는 단점이 있다     

이를 위한 대안으로 단어의 '의미'를 다차원 공간에 벡터화하는 방법      

분산 표현을 이용하여 표현된 벡터 또한 임베딩 벡터(embedding vector)라고 하며 밀집 벡터(dense vector)에도 속한다         

**'비슷한 위치에서 등장하는 단어들은 비슷한 의미를 가진다'**라는 가정하에 만들어진 표현 방법      

희소 표현이 고차원에 각 차원이 분리된 표현 방법이었다면,      

분산 표현은 저차원에 **단어의 의미를 여러 차원에다가 분산**하여 표현       

이런 표현 방법을 사용하면 **단어 간 유사도**를 계산할 수 있다       

이를 위한 학습 방법으로는 NNLM, RNNLM 등이 있으나 요즘에는 해당 방법들의 속도를 대폭 개선시킨 __Word2Vec__가 많이 쓰이고 있다

<br/>

<br/>

### Word2Vec에는 CBOW(Continuous Bag of Words)와 Skip-Gram 두 가지 방식이 있다!     

<br/>

## CBOW (Continuous Bag of Words)      

주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측하는 방법      

​      

Example)  **예문 : "The fat cat sat on the mat"**       

예측해야하는 단어 sat을 __중심 단어(center word)__ 라고 하고, 예측에 사용되는 단어들을 __주변 단어(context word)__     

__윈도우(window)__  :  중심 단어를 예측하기 위해 앞, 뒤로 볼 단어의 범위      

윈도우 크기가 n이면,  중심 단어를 예측하기 위해 참고하는 주변 단어 개수는 2n     

![image](https://wikidocs.net/images/page/22660/%EB%8B%A8%EC%96%B4.PNG)

슬라이딩 윈도우(sliding window)  :  윈도우를 계속 움직여서 주변 단어와 중심 단어 선택을 바꿔가며 학습을 위한 데이터 셋을 만드는 방법

__Word2Vec에서 입력은 모두 원-핫 벡터가 되어야 한다!__     

<br/>

![image](https://wikidocs.net/images/page/22660/word2vec_renew_1.PNG)

입력층(Input layer)  :  앞, 뒤로 윈도우 범위 안에 있는 주변 단어들의 원-핫 벡터를 입력 받음     

출력층(Output layer)  :  학습을 위해 예측하고자 하는 중간 단어의 원-핫 벡터 필요      

​     

__Word2Vec은 딥 러닝 모델(Deep Learning Model)은 아니다__     

Word2Vec는 입력층과 출력층 사이에 하나의 은닉층만 존재

Word2Vec의 은닉층은 일반적인 은닉층과 달리 활성화 함수가 존재하지 않는다!

__룩업 테이블__  연산을 담당하는 층으로 __투사층(projection layer)__ 이라고 부르기도 한다      

<br/>

![image](https://wikidocs.net/images/page/22660/word2vec_renew_2.PNG)

__M__  :  투사층의 크기

위 그림에서 __M=5__ 이기 때문에 CBOW으로 얻는 __각 단어의 임베딩 벡터의 차원은 5__      

__V__  :  단어 집합의 크기      

입력층과 투사층 사이의 가중치 __W__ 는 __V × M__ 행렬      

투사층에서 출력층 사이의 가중치 __W'__ 는 __M × V__ 행렬     

두 행렬은 동일한 행렬을 전치(transpose)한 것이 아니라, 서로 다른 행렬      



<br/>

![image](https://wikidocs.net/images/page/22660/word2vec_renew_3.PNG)

__룩업 테이블(lookup table)__      

입력 벡터와 가중치 W 행렬의 곱은      

__사실 W행렬의 i번째 행을 그대로 읽어오는 것과(lookup) 동일__        

__훈련 전에 W와 W'는 대게 굉장히 작은 랜덤 값을 가진다__      

__CBOW는 주변 단어로 중심 단어를 더 정확히 맞추기 위해 계속해서 이 W와 W'를 학습해가는 구조__             

->  __이유  : lookup해온 W의 각 행벡터가 사실 Word2Vec을 수행한 후의 각 단어의 M차원의 크기를 갖는 임베딩 벡터들__

<br/>

![image](https://wikidocs.net/images/page/22660/word2vec_renew_4.PNG)

각 주변 단어의 원-핫 벡터에 대해서 가중치 W가 곱해서 생겨진 결과 벡터들은 투사층에서 만나 이 벡터들의 평균인 벡터를 구하게 된다     

<br/>

![image](https://wikidocs.net/images/page/22660/word2vec_renew_5.PNG)

구해진 평균 벡터는 두번째 가중치 행렬 W'와 곱해진다      

곱셈의 결과로 원-핫 벡터들과 차원이 V로 동일한 벡터가 나온다      

이 벡터에 CBOW는 softmax 함수를 취한다

이렇게 나온 벡터를 __스코어 벡터(score vector)__ 라고 한다     

스코어 벡터의 __j번째 인덱스가 가진 0과 1사이의 값은 j번째 단어가 중심 단어일 확률__     

스코어 벡터를 𝑦^,  중심 단어를 y라고 했을 때, 이 두 벡터값의 오차를 줄이기위해       

CBOW는 loss function으로 cross-entropy 함수를 사용한다     

![image](https://wikidocs.net/images/page/22660/crossentrophy2.PNG)

y가 원-핫 벡터라는 점을 고려하면, 위와 같이 간소화시킬 수 있다     

__y^가 y를 정확하게 예측한 경우,  -1 log(1) = 0이 되기 때문에 결과적으로 cross-entropy의 값은 0이 된다__       

역전파(Back Propagation)를 수행하면 W와 W'가 학습이 된다      

학습이 다 되었다면 M차원의 크기를 갖는 W의 행이나 W'의 열로부터 어떤 것을 임베딩 벡터로 사용할지를 결정하면 된다       

때로는 W와 W'의 평균치를 가지고 임베딩 벡터를 선택하기도 한다       

<br/>

<br/>

-------

<br/>

## Skip-gram      

CBOW와 메커니즘 자체는 동일하다      

CBOW에서는 주변 단어를 통해 중심 단어를 예측했다면, Skip-gram은 중심 단어에서 주변 단어를 예측

​     

Example) 

![image](https://wikidocs.net/images/page/22660/skipgram_dataset.PNG)

![image](https://wikidocs.net/images/page/22660/word2vec_renew_6.PNG)

__중심 단어에 대해서 주변 단어를 예측하므로 투사층에서 벡터들의 평균을 구하는 과정은 없다__     

전반적으로 Skip-gram이 CBOW보다 성능이 좋다고 알려져있다       

<br/>

<br/>

-----

<br/>

## NNLM  vs Word2Vec     

![image](https://wikidocs.net/images/page/22660/word2vec_renew_7.PNG)

NNLM은 단어 간 유사도를 구할 수 있도록 워드 임베딩의 개념을 도입      

NNLM의 느린 학습 속도와 정확도를 개선하여 탄생한 것이 Word2Vec        

​      

NNLM은 언어 모델이므로 다음 단어를 예측하지만,       

Word2Vec(CBOW)은 워드 임베딩 자체가 목적이므로 다음 단어가 아닌 중심 단어를 예측하게 하여 학습       

​      

중심 단어를 예측하게 하므로서 NNLM이 예측 단어의 이전 단어들만을 참고하였던 것과는 달리,      

Word2Vec은 예측 단어의 전, 후 단어들을 모두 참고      

​      

Word2Vec은 NNLM에 존재하던 활성화 함수가 있는 은닉층 제거

​      

Word2Vec이 보다 학습 속도에서 강점을 가지는 이유는 은닉층을 제거한 것뿐만 아니라 추가적으로 사용되는 기법들 덕분       

대표적인 기법으로 계층적 소프트맥스(hierarchical softmax)와 네거티브 샘플링(negative sampling)이 있다       

NNLM의 연산량: ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%28n%20%5Ctimes%20m%29%20&plus;%20%28n%20%5Ctimes%20m%20%5Ctimes%20h%29%20&plus;%20%28h%5Ctimes%20V%29)

Word2Vec은 출력층에서의 연산에서 V를 log(V)로 바꿀 수 있다      

이에 따라 Word2Vec의 연산량은 NNLM보다 배는 빠른 학습 속도를 가진다

Word2Vec의 연산량: ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%28n%20%5Ctimes%20m%29%20&plus;%20%28m%20%5Ctimes%20log%28V%29%29)

  

<br/>

<br/>

-------

<br/>

## 사전 훈련된 Word2Vec 임베딩 (Pre-trained Word2Vec embedding) 소개      

위키피디아 등의 방대한 데이터로 사전에 훈련된 워드 임베딩(pre-trained word embedding vector)를 가지고 와서 해당 벡터들의 값을 원하는 작업에 사용 할 수도 있다     

#### 영어      

구글이 제공하는 사전 훈련된 Word2Vec 모델을 사용하는 방법      

구글은 사전 훈련된 3백만 개의 Word2Vec 단어 벡터들을 제공       

각 임베딩 벡터의 차원은 300     

gensim을 통해서 이 모델을 불러오는 건 매우 간단

​      

모델 다운로드 경로:     

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit      

압축 파일의 용량 약 1.5GB    파일의 압축을 풀면 약 3.3GB       

```python
import gensim

# 구글의 사전 훈련된 Word2Vec 모델을 로드합니다.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin 파일 경로', binary=True)  

print(model.vectors.shape) # 모델의 크기 확인
(3000000, 300)
# 모델의 크기는 3,000,000 x 300. 즉, 3백만 개의 단어와 각 단어의 차원은 300
# 파일의 크기가 3기가가 넘는 이유를 계산해보면 아래와 같다.
# 3 million words * 300 features * 4bytes/feature = ~3.35GB

print (model.similarity('this', 'is')) # 두 단어의 유사도 계산하기
print (model.similarity('post', 'book'))

0.407970363878
0.0572043891977

print(model['book']) # 단어 'book'의 벡터 출력

[ 0.11279297 -0.02612305 -0.04492188  0.06982422  0.140625    0.03039551
 -0.04370117  0.24511719  0.08740234 -0.05053711  0.23144531 -0.07470703
... 300개의 값이 출력되는 관계로 중략 ...
  0.03637695 -0.16796875 -0.01483154  0.09667969 -0.05761719 -0.00515747]
```

<br/>

#### 한국어      

한국어의 미리 학습된 Word2Vec 모델은 https://github.com/Kyubyong/wordvectors 에 공개되어있다       

모델 다운로드 경로 : 

https://drive.google.com/file/d/0B0ZXk88koS2KbDhXdWg1Q2RydlU/view        

링크로부터 77MB 크기의 ko.zip 파일을 다운로드 받아 압축을 풀면 ko.bin라는 50MB 파일이 있다        

```python
import gensim
model = gensim.models.Word2Vec.load('ko.bin 파일의 경로')

result = model.wv.most_similar("강아지")
print(result)

[('고양이', 0.7290453314781189), ('거위', 0.7185634970664978), ('토끼', 0.7056223750114441), ('멧돼지', 0.6950401067733765), ('엄마', 0.693433403968811), ('난쟁이', 0.6806551218032837), ('한마리', 0.6770296096801758), ('아가씨', 0.675035297870636), ('아빠', 0.6729634404182434), ('목걸이', 0.6512461304664612)]
```

<br/>

<br/>

------

#### 형태소 분석기는 Mecab이 제일 빠르다 

https://github.com/SOMJANG/Mecab-ko-for-Google-Colab 

------------

<br/>

<br/>

## SGNS, Skip-Gram with Negative Sampling     

​      

### Negative Sampling      

__Word2Vec이 학습 과정에서 전체 단어 집합이 아니라 일부 단어 집합에만 집중할 수 있도록 하는 방법__

만약 현재 집중하고 있는 중심 단어와 주변 단어가 '강아지'와 '고양이', '귀여운'과 같은 단어라면,      

이 단어들과 __별 연관 관계가 없는__ '돈가스'나 '컴퓨터'와 같은 __수많은 단어의 임베딩 벡터값까지 업데이트하는 것은 비효율적__

현재 집중하고 있는 주변 단어가 '고양이', '귀여운'      

여기에 '돈가스', '컴퓨터', '회의실'과 같은 단어 집합에서 __무작위로 선택된 주변 단어가 아닌 단어들을 일부 가져온다__     

이렇게 하나의 중심 단어에 대해서 __전체 단어 집합보다 훨씬 작은 단어 집합을 만들어놓고 마지막 단계를 이진 분류 문제로 변환__       

주변 단어들을 긍정(positive), 랜덤으로 샘플링 된 단어들을 부정(negative)으로 레이블링한다면 이진 분류 문제를 위한 데이터셋이 된다         

이는 기존의 단어 집합의 크기만큼의 선택지를 두고 다중 클래스 분류 문제를 풀던 Word2Vec보다 훨씬 연산량에서 효율적        

<br/>

<br/>

## SGNS      

![image](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC1.PNG)

![image](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC1-1.PNG)

Skip-gram은 중심 단어로부터 주변 단어를 예측하는 모델

<br/>

SGNS는 이와는 다른 접근 방식

__중심 단어와 주변 단어가 모두 입력이 되고, 이 두 단어가 실제로 윈도우 크기 내에 존재하는 이웃 관계인지 그 확률을 예측__       

![image](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC1-2.PNG)

<br/>

__기존의 Skip-gram 데이터셋을 SGNS의 데이터셋으로 바꾸는 과정__

![image](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC3.PNG)

좌측의 테이블은 기존의 Skip-gram을 학습하기 위한 데이터셋       

__Skip-gram은 기본적으로 중심 단어를 입력, 주변 단어를 레이블__        

​        

기존의 Skip-gram 데이터셋에서 중심 단어와 주변 단어를 각각 입력1, 입력2로      

이 둘은 실제로 윈도우 크기 내에서 __이웃 관계였므로 레이블은 1__ 로        

이제 레이블이 0인 샘플들을 준비할 차례

![image](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC4.PNG)

__단어 집합에서 랜덤으로 선택한 단어들__ 을 입력2로 하고, __레이블을 0__ 으로     

​       

__두 개의 임베딩 테이블__ 준비       

두 임베딩 테이블은 훈련 데이터의 단어 집합의 크기를 가지므로 크기가 같다       

![image](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC5.PNG)

하나는 입력 1인 __중심 단어의 테이블 룩업__ 을 위한 임베딩 테이블     

하나는 입력 2인 __주변 단어의 테이블 룩업__ 을 위한 임베딩 테이블     

__각 단어__ 는 각 임베딩 테이블을 테이블 룩업하여 __임베딩 벡터__ 로 변환

![image](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC6.PNG)

​     

그 후의 연산     

![image](https://wikidocs.net/images/page/69141/%EA%B7%B8%EB%A6%BC7.PNG)

중심 단어와 주변 단어의 내적값을 이 모델의 예측값으로      

레이블과의 오차로부터 역전파하여 중심 단어와 주변 단어의 임베딩 벡터값을 업데이트