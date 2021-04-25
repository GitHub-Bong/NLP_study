

# 딥 러닝을 이용한 자연어 처리 입문

<br/>

[관련논문](https://www.aclweb.org/anthology/N18-1202/)            

## ELMo (Embeddings from Language Model, ELMo)      

<br/>

2018년에 제안된 새로운 워드 임베딩 방법론       

'언어 모델로 하는 임베딩'      

> 언어 모델은 오랜 기간 동안 모델 그 자체로는 별다른 실용적 가치를 갖지 못하고 있었다. 어떤 언어 모델이 아무리 잘 학습되었다고 한 들, 그럴듯한 문장을 만들어내는 것 이외에는 딱히 사용할 만한 곳을 찾기 힘들었기 때문이다. 언어 모델보다는 확실한 실용적 가치를 기대할 수 있는 기계 번역, 질의 응답이 인기를 끈 것도 비슷한 맥락이다. 그런데 2018년 들어, 언어의 전반적인 특징을 학습한 언어 모델을 다양한 자연어 처리 과제에 활용할 경우 큰 폭의 성능 향상을 보인다는 결과들이 보고되며, 언어 모델은 이제 실용적 가치를 갖기 시작했다. 다양한 언어의 특성을 학습한 pre-trained 언어 모델의 knowledge transfer 하면, 내가 목표하고자 하는 다른 자연어 처리 과제에 효과적이라는 것이다.    

ELMo의 가장 큰 특징          

**사전 훈련된 언어 모델(Pre-trained language model)** 을 사용!        

<br/>       

### 1. ELMo (Embeddings from Language Model)       

Bank Account(은행 계좌)와 River Bank(강둑)에서의 Bank는 전혀 다른 의미!       

Word2Vec이나 GloVe 등으로 표현된 임베딩 벡터들은 이를 제대로 반영하지 못한다         

단어를 임베딩하기 전에 전체 문장을 고려해서 임베딩을 하겠다!        

그래서 탄생한 것이 **문맥을 반영한 워드 임베딩(Contextualized Word Embedding)**        

<br/>

### 2. biLM (Bidirectional Lanuage Model)의 사전 훈련     

ELMo에서 말하는 biLM은 기본적으로 다층 구조(Multi-layer)를 전제       

은닉층이 최소 2개 이상이라는 의미        

ELMo는 순방향 RNN 뿐만 아니라, 반대 방향으로 문장을 스캔하는 역방향 RNN 또한 활용      

ELMo는 양쪽 방향의 언어 모델을 둘 다 활용한다고하여 **biLM(Bidirectional Language Model)**      

​        ![image](https://wikidocs.net/images/page/33930/forwardbackwordlm2.PNG)

은닉층이 2개인 순방향 언어 모델과 역방향 언어 모델의 모습        

​           

biLM의 입력이 되는 워드 임베딩 방법  : char CNN

글자(character) 단위로 계산되는데, 이렇게 하면 마치 서브단어(subword)의 정보를 참고하는 것처럼 문맥과 상관없이 dog란 단어와 doggy란 단어의 연관성을 찾아낼 수 있고 OOV에도 견고한다는 장점!        

​       

주의할 점 :        

**양방향 RNN**과 ELMo에서의 **biLM**은 다소 다르다!       

양방향 RNN은 순방향 RNN의 은닉 상태와 역방향의 RNN의 은닉 상태를 다음 층의 입력으로 보내기 전에 연결(concatenate)       

biLM의 순방향 언어모델과 역방향 언어모델이 각각의 은닉 상태만을 다음 은닉층으로 보내며 훈련시킨 후에 ELMo 표현으로 사용하기 위해서 은닉 상태를 연결(concatenate)시키는 것과 다르다! 

<br/>

 

###  3. biLM의 활용      

play란 단어가 임베딩이 되고 있다는 가정       

![image](https://wikidocs.net/images/page/33930/playwordvector.PNG) 

play라는 단어를 임베딩 하기위해 ELMo는 점선의 사각형 내부의 각 층의 결과값을 재료로 사용       

즉, 해당 시점(time-step)의 BiLM의 각 층의 출력값을 가져온다        

그리고 순방향 언어 모델과 역방향 언어 모델의 각 층의 출력값을 연결(concatenate)하고 추가 작업 진행       

각 층의 출력값이란 첫번째는 임베딩 층을 말하며, 나머지 층은 각 층의 은닉 상태        

ELMo의 직관적인 아이디어         각 층의 출력값이 가진 정보는 전부 서로 다른 종류의 정보를 갖고 있을 것이므로, 이들을 모두 활용한다!        

__ELMo가 임베딩 벡터를 얻는 과정__       

__1) 각 층의 출력값을 연결(concatenate)__

![image](https://wikidocs.net/images/page/33930/concatenate.PNG)

__2) 각 층의 출력값 별로 가중치를 준다__

![image](https://wikidocs.net/images/page/33930/weight.PNG)

__3) 각 층의 출력값을 모두 더한다__ 

![image](https://wikidocs.net/images/page/33930/weightedsum.PNG)

2)번과 3)번의 단계를 요약하여 '가중합(Weighted Sum)을 한다' 

__4) 벡터의 크기를 결정하는 스칼라 매개변수를 곱한다__

![image](https://wikidocs.net/images/page/33930/scalarparameter.PNG)

​       

이렇게 완성된 벡터       __ELMo 표현(representation)__       

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbafJAN%2FbtqEVVNXUkp%2FUofpBtpTxC0Mjk8dBjTe40%2Fimg.png)

<br/>

예를 들어 텍스트 분류 작업을 하고 싶다고 가정. 그렇다면 ELMo 표현을 어떻게 텍스트 분류 작업에 사용할 수 있을까?       

ELMo 표현은 기존의 임베딩 벡터와 함께 사용할 수 있다!      

텍스트 분류 작업을 위해 GloVe와 같은 기존의 방법론을 사용한 임베딩 벡터 준비      

준비된 ELMo 표현을 GloVe 임베딩 벡터와 연결(concatenate)해서 입력으로 사용할 수 있다!       

이 때, ELMo 표현을 만드는데 사용되는 사전 훈련된 언어 모델의 가중치는 고정. 그리고 사용한 s1,  s2,  s3와 γ는 훈련 과정에서 학습된다      

![image](https://wikidocs.net/images/page/33930/elmorepresentation.PNG)

ELMo 표현이 기존의 GloVe 등과 같은 임베딩 벡터와 함께 NLP 태스크의 입력이 되는 모습     

<br/>

<br/>

### 4. ELMo 표현을 사용해서 스팸 메일 분류하기      

```python
%tensorflow_version 1.x  #텐서플로우 버전을 1버전으로 설정
pip install tensorflow-hub  #텐서플로우 허브 설치

import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
import urllib.request
import pandas as pd
import numpy as np

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
# 텐서플로우 허브로부터 ELMo를 다운로드

sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
```

[파일 원본 출처](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

```python
# 스팸 메일 분류하기 데이터 다운로드

urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding='latin-1')
data[:5]
```

![image](https://wikidocs.net/images/page/22894/%ED%9B%88%EB%A0%A8%EB%8D%B0%EC%9D%B4%ED%84%B0.PNG)

```python
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
y_data = list(data['v1'])
X_data = list(data['v2'])

X_data[:5]
['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
 'Ok lar... Joking wif u oni...',
 "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
 'U dun say so early hor... U c already then say...',
 "Nah I don't think he goes to usf, he lives around here though"]
print(y_data[:5])
[0, 0, 1, 0, 0]
```

​       

```python
print(len(X_data))
n_of_train = int(len(X_data) * 0.8)
n_of_test = int(len(X_data) - n_of_train)
print(n_of_train)
print(n_of_test)
5572
4457
1115
#훈련 데이터와 테스트 데이터를 8:2 비율로 분할
X_train = np.asarray(X_data[:n_of_train]) #X_data 데이터 중에서 앞의 4457개의 데이터만 저장
y_train = np.asarray(y_data[:n_of_train]) #y_data 데이터 중에서 앞의 4457개의 데이터만 저장
X_test = np.asarray(X_data[n_of_train:]) #X_data 데이터 중에서 뒤의 1115개의 데이터만 저장
y_test = np.asarray(y_data[n_of_train:]) #y_data 데이터 중에서 뒤의 1115개의 데이터만 저장
```

​       

__ELMo는 텐서플로우 허브로부터 가져온 것이기 때문에 케라스에서 사용하기 위해서는 케라스에서 사용할 수 있도록 변환해주는 작업들이 필요__       

```python
def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]
# 데이터의 이동이 케라스 → 텐서플로우 → 케라스가 되도록 하는 함수
```

```python
from keras.models import Model
from keras.layers import Dense, Lambda, Input

input_text = Input(shape=(1,), dtype=tf.string)
embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
hidden_layer = Dense(256, activation='relu')(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)
model = Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

```python
history = model.fit(X_train, y_train, epochs=1, batch_size=60)
Epoch 1/1
4457/4457 [==============================] - 1508s 338ms/step - loss: 0.1129 - acc: 0.9619

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
1115/1115 [==============================] - 381s 342ms/step
테스트 정확도: 0.9803
```

