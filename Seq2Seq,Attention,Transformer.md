# Seq2Seq / Attention / Transformer  정리        

- [Seq2Seq](#Seq2Seq) 
  - [Encoder](#인코더)
  - [Decoder](#디코더)
- [Attention](#Attention)
  - [Attention Value 구하기](#Attention-Mechanism)  
- [Transformer](#Transformer)
  - [Positional Encoding](#Positional-Encoding)
  - [트랜스포머에서 사용되는 세가지 어텐션](#트랜스포머에서-사용되는-세가지-어텐션)
  - [Encoder](#인코더)
    - [Multi-head Self-Attention](#Multi-head-Self-Attention)
    - [Position-wise Feed Foward Neural Network](#Position-wise-Feed-Foward-Neural-Network)
    - [Add & Norm](#Add--Norm)
  - [Decoder](#디코더)
    - [Masked Multi-head Self-Attention](#Masked-Multi-head-Self-Attention)
    - [Multi-head Attention(Encoder-Decoder Attention)](#Multi-head-AttentionEncoder-Decoder-Attention)
- [Transformer 요약](#Transformer-요약)
  - [embedding + Positional Encoding?](#embedding--Positional-Encoding)
  - [2가지 Self-Attention](#2가지-Self-Attention)
  - [두 Self-Attention의 차이](#두-Self-Attention의-차이)
  - [Multi-head Attention(Encoder-Decoder Attention)](#Multi-head-AttentionEncoder-Decoder-Attention)
  - [3가지 Multi-head Attention](3가지-Multi-head-Attention)
  - [Position-wise FFNN](#Position-wise-FFNN)
  - [Add & Norm](#Add--Norm)



## Seq2Seq         

__인코더와 디코더로 구성!__       

### 인코더       

입력 문장이 단어 토큰화를 통해 단어 단위로 쪼개져 각각의 단어 토큰들이 입력이 됨.     

순차적으로 입력받은 뒤에 마지막 시점의 은닉 상태 출력 이것이      

이 모든 단어 정보들을 압축한 __컨텍스트(context vector) 벡터!__         



### 디코더                  

컨텍스트 벡터을 디코더 RNN 셀의 첫번째 은닉 상태로 사용함.    

디코더는 초기 입력으로 '<sos>' 사용함.      

이로 부터 다음에 등장학 확률이 가장 높은 단어 예측함.     

* 테스트 과정     
  * 다음 단어로 je를 예측했으면 je를 다음 시점의 RNN 셀의 입력으로 사용     
  * 이런 식으로 다음 단어 예측하고, 그 예측 단어를 다음 시점의 입력으로 넣음.     
  * '<eos>' 가 다음 단어로 예측될 때까지 반복됨     
* 훈련 과정     
  * 정답을 알려주면서 훈련   ( __교사강요 teacher forcing__ )          



<br/>

Seq2Seq 모델에는 두 가지 문제 발생     

__하나의 고정된 크기의 벡터에 모든 정보를 압축하다보니 정보 손실 발생__     

__RNN 고질적인 문제인 기울기 손실(Vanishing Gradient) 문제 발생__      

​      

<br/>



## Attention      

디코더에서 출력 단어를 예측하는 매 시점마다     

인코더에서 해당 시점에서 예측해야할 단어와 연관이 있는 입력 단어 부분을 참고한다!      



어텐션 메커니즘에서는 시점 t에서 출력 단어를 예측하기 위해      

t-1 시점의 은닉 상태와 이전 시점 t-1에서 나온 출력 단어  그리고 __Attention Value__ 가 필요하다!        

​        

### Attention Mechanism        



__1) Attention Score 구하기__      

현재 t 시점에서 단어를 예측하기 위해       

인코더의 모든 은닉 상태 각각이 디코더의 현 시점의 은닉 상태 와 얼마나 유사한지를 판단하는 스코어값       

Ex) dot-product attention 에서는 디코더의 t 시점 은닉 상태와 인코더의 각 은닉 상태를 내적    

​     

__2) softmax 함수를 통해 Attention Distribution 구하기__      

위에서 나온 어텐션 스코어의 모음값에 softmax 함수 적용한 것을 Attention Distribution이라 함     

각각의 값은 Attention Weight 라 함      

​       

__3) 각 Attention Weight 와 은닉 상태를 가중합하여 Attention Value  구하기__        

각 인코더의 은닉 상태와 어텐션 가중치 값들을 곱하고, 최종적으로 모두 더한다  (= 가중합)     

![image](https://latex.codecogs.com/gif.latex?a_%7Bt%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Calpha%20_%7Bi%7D%5E%7Bt%7Dh_%7Bi%7D)

 이것이 바로 __Attention Value__   ![image](https://latex.codecogs.com/gif.latex?a_%7Bt%7D)          

이러한 Attention Value 를 context vector 라고도 부름.  (Seq2Seq에서는 인코더의 마지막 은닉 상태를 context vector 라고 불렀음)      

​      

Attention Value를 구하면  어텐션 메커니즘은       

__디코더의 현 시점 은닉 상태와 Attention Value를 결합(concatenate)해 하나의 벡터로 만든다!__      

이를 예측 연산의 입력으로 사용함      

​     

논문에서는 결합한 하나의 벡터를 출력층으로 보내기 전에        

가중치 행렬과 곱한 후 하이퍼볼릭탄젠트 함수를 지나도록  신경망 연산을 거침      

![image](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bs%7D_%7Bt%7D%20%3D%20%5Ctanh%28%5Cmathbf%7BW_%7Bc%7D%7D%5B%7Ba%7D_t%3B%7Bs%7D_t%5D%20&plus;%20b_%7Bc%7D%29)          

Seq2Seq 에서는 출력층의 입력이 t시점 은닉상태 St였는데     

어텐션 메커니즘에서는 ![image](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bs%7D_%7Bt%7D)       

​      

최종적으로    ![image](https://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D_t%20%3D%20%5Ctext%7BSoftmax%7D%5Cleft%28%20W_y%5Ctilde%7Bs%7D_t%20&plus;%20b_y%20%5Cright%29)       예측 벡터를 얻는다!        

<br/>

<br/>



## Transformer       

기존 Seq2Seq의 인코더-디코더 구조를 따르면서도, Attention 만으로 구현한 모델     

Seq2Seq 구조에서는 인코더와 디코더에서 각각 하나의 RNN이 t개의 시점을 가지는 구조였다면      

Transformer는 인코더와 디코더라는 단위가 N개로 구성되는 구조!      

(논문에서는 인코더와 디코더의 개수를 각각 6개 사용)     

![image](https://wikidocs.net/images/page/31379/transformer2.PNG)     

 디코더는 기존 Seq2Seq 구조처럼 '<sos>'를 입력으로 받아 '<eos>' 가 나올 때까지 연산 진행     

​      

RNN 은 단어를 순차적으로 입력받아 각 단어의 위치정보를 가질 수 있었다     

Transformer는 단어 입력을 순차적으로 받는 방식이 아니므로 단어의 위치 정보를 다른 방식으로 알려줄 필요가 있다.    

그래서 Transformer의 인코더와 디코더는 단순히 각 단어의 임베딩 벡터들을 입력받는 것이 아니라 임베딩 벡터에서 조정된 값을 입력받는다.     

​         

### Positional Encoding     

Transformer는 각 단어의 임베딩 벡터에 위치 정보들을 더하여 모델의 입력으로 사용하는데 이를 __positional encoding__      

![image](https://wikidocs.net/images/page/31379/transformer7.PNG)      

![image](https://latex.codecogs.com/gif.latex?PE_%7B%28pos%2C%5C%202i%29%7D%3Dsin%28pos/10000%5E%7B2i/d_%7Bmodel%7D%7D%29)       

![image](https://latex.codecogs.com/gif.latex?PE_%7B%28pos%2C%5C%202i&plus;1%29%7D%3Dcos%28pos/10000%5E%7B2i/d_%7Bmodel%7D%7D%29)      

pos  :  입력 문장에서 임베딩 벡터의 위치       

i  :  임베딩 벡터 내의 차원의 인덱스    (각 차원의 인덱스가 짝수면 사인함수, 홀수면 코사인 함수)      

dmodel  :  모든 층의 출력 차원  (논문에서는 512)     

같은 단어라 하더라도 문장 내 위치에 따라 입력 벡터 값이 달라짐       

Transformer의 입력은 순서 정보가 고려된 임베딩 벡터다!        

​        

### 트랜스포머에서 사용되는 세가지 어텐션      

![image](https://wikidocs.net/images/page/31379/attention.PNG)       

![image](https://wikidocs.net/images/page/31379/transformer_attention_overview.PNG)  



Encoder-Decoder Attention은 Query가 디코더의 벡터, Key, Value가 인코더의 벡터    

__Multi-head__    어텐션을 병렬적으로 수행하는 방법      

<br/>

### 인코더     

하나의 인코더는 크게 총 2개의 sublayers로 구성      

__Multi-head Self-Attention__ , __Position-wise Feed Foward Neural Network__     

​         

#### Multi-head Self-Attention        

__(1) 인코더의 셀프 어텐션__      

seq2seq의 어텐션 경우     

Q = Query : t 시점의 디코더 셀에서의 은닉 상태        

K = Keys : 모든 시점의 인코더 셀의 은닉 상태들       

V = Values : 모든 시점의 인코더 셀의 은닉 상태들         

​      

transformer의 셀프 어텐션 경우      

Q : 입력 문장의 모든 단어 벡터들       

K : 입력 문장의 모든 단어 벡터들       

V : 입력 문장의 모든 단어 벡터들        

​      

Q, K, V 벡터들은 인코더의 초기 입력인 512 차원을 가지는 단어 벡터들보다 더 작은 64 차원을 가진다      

![image](https://latex.codecogs.com/gif.latex?d_%7Bmodel%7D%20/%20num%5C%3A%20%5C%3A%20heads)    논문에서 512 / 8 = 64           

​       

transformer에서 __셀프 어텐션 동작 메커니즘__      

__Q, K, V 벡터 얻기__      

![image](https://wikidocs.net/images/page/31379/transformer11.PNG)     

각 가중치 행렬은 ![image](https://latex.codecogs.com/gif.latex?d_%7Bmodel%7D%20%5Ctimes%20%28d_%7Bmodel%7D/num%20%5C%3A%20%5C%3A%20heads%29)   크기를 가지며 훈련 과정에서 학습 된다     

모든 단어 벡터에 위와 같은 과정을 거치면 각각의 Q, K, V 벡터를 얻는다     

​      

각 Q 벡터마다 모든 K 벡터에 대해 __어텐션 스코어__ 구한다       

-> __어텐션 분포__ 를 구한 뒤 이를 이용해        

-> 모든 V 벡터를 __가중합__ 하여 __어텐션 값 또는 컨텍스트 벡터__ 를 구한다         

이를 모든 Q 벡터에 대해 반복한다      

​        

__Transformer 에서는 ![image](https://latex.codecogs.com/gif.latex?score%28q%2C%20k%29%3Dq%5Ccdot%20k) 가 아니라      ![image](https://latex.codecogs.com/gif.latex?score%28q%2C%20k%29%3Dq%5Ccdot%20k/%5Csqrt%7Bn%7D)  사용__      

이를 __Scaled dot-product Attention__ 이라 함    

![image](https://wikidocs.net/images/page/31379/transformer13.PNG)      

​        

그 다음, __Attention Score__ 에 softmax 함수를 사용해       

__Attention Distribution__ 을 구하고        

각 V 벡터와 가중합해 __Attention Value__ 을 구한다     

이를 I에 대한 __어텐션 값__  또는 __context vector__  라고 함   

각 Q 벡터에 대해서 모두 동일한 과정을 반복해 각각에 대한 어텐션 값을 구한다  

![image](https://wikidocs.net/images/page/31379/transformer14_final.PNG)        





<br/>

__사실 행렬 연산을 사용하면 일괄 계산이 가능해진다__      

__실제로도 행렬 연산으로 구현된다__       

​       

![image](https://wikidocs.net/images/page/31379/transformer12.PNG)   

문장 행렬에 가중치 행렬을 곱해 Q 행렬, K 행렬, V 행렬을 구한다     

​        



![image](https://wikidocs.net/images/page/31379/transformer15.PNG)  

결과 행렬: 각각의 단어의 Q 벡터와 K 벡터의 내적이 각 행렬의 원소가 되는 행렬     

결과 행렬 값에 전체적으로 ![image](https://latex.codecogs.com/gif.latex?%5Csqrt%7Bd_%7Bk%7D%7D) 을 나눠주면 __Attention Score__ 행렬이 된다     

​        

  ![image](https://wikidocs.net/images/page/31379/transformer16.PNG)

__Attention Score 행렬__ 에 softmax 함수 사용하고 V 행렬 곱해 __Attention Value 행렬__  구한다     



 ![image](https://latex.codecogs.com/gif.latex?Attention%28Q%2C%20K%2C%20V%29%20%3D%20softmax%28%7BQK%5ET%5Cover%7B%5Csqrt%7Bd_k%7D%7D%7D%29V) 

​       

__정리__     

입력 문장의 길이 : seq_len    

문장 행렬의 크기 : ![image](https://latex.codecogs.com/gif.latex?%28%7Bseq%5C%3A%20len%7D%2C%5C%20d_%7Bmodel%7D%29)       

Q 행렬, K 행렬의 크기 : ![image](https://latex.codecogs.com/gif.latex?%28%7Bseq%5C%3A%20len%7D%2C%5C%20d_%7Bk%7D%29)

V 행렬의 크기 : ![image](https://latex.codecogs.com/gif.latex?%28%7Bseq%5C%3A%20len%7D%2C%5C%20d_%7Bv%7D%29)

논문에서는 ![image](https://latex.codecogs.com/gif.latex?d_%7Bmodel%7D%20/%20num%5C%3A%20heads%3D%20d_%7Bk%7D%3Dd_%7Bv%7D) 

가중치 행렬의 크기 : ![image](https://latex.codecogs.com/gif.latex?%28d_%7Bmodel%7D%2C%5C%20d_%7Bk%7D%29%2C%20%5C%3A%20%28d_%7Bmodel%7D%2C%5C%20d_%7Bv%7D%29)         

​     

__Attention Value 행렬의 크기__ : ![image](https://latex.codecogs.com/gif.latex?%28%7Bseq%5C%3A%20len%7D%2C%5C%20d_%7Bv%7D%29)

<br/>     

![image](https://wikidocs.net/images/page/31379/transformer17.PNG)

![image](https://latex.codecogs.com/gif.latex?num%5C%3A%20%5C%3A%20heads) 의 의미        

한 번의 어텐션을 하는 것 보다 여러 번의 어텐션을 병렬로 사용하는 것이 더 효과적      

![image](https://latex.codecogs.com/gif.latex?d_%7Bmodel%7D%20/%5C%2C%20num%20%5C%3A%20%5C%3A%20heads) 차원을 가지는 Q, K, V에 대해서       

![image](https://latex.codecogs.com/gif.latex?num%5C%3A%20%5C%3A%20heads) 개의 병렬 어텐션 수행        

논문에서는 8개의 병렬 어텐션       

​        

![image](https://wikidocs.net/images/page/31379/transformer18_final.PNG)       

모든 어텐션 헤드를 연결한 어텐션 헤드 행렬의 크기 ![image](https://latex.codecogs.com/gif.latex?%28seq%5C%3A%20%5C%3A%20len%2C%5C%20d_%7Bmodel%7D%29)        

​        

![image](https://wikidocs.net/images/page/31379/transformer19.PNG) 

어테션 헤드를 모두 연결한 행렬에 또 다른 가중치 행렬을 곱해 나온 결과가      

__Multi-head attention matrix의 최종 결과 행렬__         

__인코더의 입력이었던 문장 행렬의 크기 ![image](https://latex.codecogs.com/gif.latex?%28seq%5C%3A%20%5C%3A%20len%2C%5C%20d_%7Bmodel%7D%29)   와 동일__      



Transformer는 다수의 인코더를 쌓은 형태이기 때문에 __인코더에서의 입력의 크기가 출력에서도 동일 크기로 계속 유지되어야만__  다음 인코더에서도 다시 입력이 될 수 있다          

​           

#### Position-wise Feed Foward Neural Network         



__Position-wise FFNN__       

쉽게 생각하면 완전 FFNN(Fully-connected FFNN)      

![image](https://latex.codecogs.com/gif.latex?FFNN%28x%29%20%3D%20MAX%280%2C%20x%7BW_%7B1%7D%7D%20&plus;%20b_%7B1%7D%29%7BW_2%7D%20&plus;%20b_2)  

![image](https://wikidocs.net/images/page/31379/positionwiseffnn.PNG)  

```python
outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
outputs = tf.keras.layers.Dense(units=d_model)(outputs)
```

여기서 x는 Multi-head attention 결과 나온 ![image](https://latex.codecogs.com/gif.latex?%28seq%5C%3A%20%5C%3A%20len%2C%5C%20d_%7Bmodel%7D%29) 크기 행렬       

![image](https://latex.codecogs.com/gif.latex?W_1) 의 크기  

![image](https://latex.codecogs.com/gif.latex?%28d_%7Bmodel%7D%2C%5C%20d_%7Bff%7D%29)       

![image](https://latex.codecogs.com/gif.latex?W_2) 의 크기

![image](https://latex.codecogs.com/gif.latex?%28d_%7Bff%7D%2C%5C%20d_%7Bmodel%7D%29)           

논문에서는 ![image](https://latex.codecogs.com/gif.latex?d_%7Bff%7D) 의 크기 2048      

​             

#### Add & Norm       

__두번째 서브층을 지난 인코더의 최종 출력은 여전히 인코더의 입력 크기__ 였던  ![image](https://latex.codecogs.com/gif.latex?%28seq%5C%3A%20%5C%3A%20len%2C%5C%20d_%7Bmodel%7D%29)          

​             

​       

![image](https://wikidocs.net/images/page/31379/transformer21.PNG)

인코더에 추가적으로 사용하는 기법 __Add & Norm__       

__잔차 연결(residual connection)__ 과 __층 정규화(layer normalization)__        

​      

__잔차 연결__    

![image](https://wikidocs.net/images/page/31379/transformer22.PNG) 

F(x) 가 Transformer에서는 서브층     

즉, 잔차 연결은 서브층의 입력과 출력을 더하는 것      

서브층의 입력과 출력은 동일한 차원을 갖고 있으므로, 둘은 덧셈 연산 가능      

![image](https://latex.codecogs.com/gif.latex?x&plus;Sublayer%28x%29)       

​       

__층 정규화__       

![image](https://latex.codecogs.com/gif.latex?LN%20%3D%20LayerNorm%28x&plus;Sublayer%28x%29%29)        

잔차 연결을 거친 결과는 이어서 층 정규화 거친다         

![image](https://wikidocs.net/images/page/31379/layer_norm_new_2_final.PNG)

층 정규화를 위해서 화살표 방향으로 각각 평균과 분산을 구한다     

 ![image](https://latex.codecogs.com/gif.latex?ln_%7Bi%7D%20%3D%20LayerNorm%28x_%7Bi%7D%29)        

각 화살표 방향의 벡터 ![image](https://latex.codecogs.com/gif.latex?x_%7Bi%7D) 는      

![image](https://latex.codecogs.com/gif.latex?ln_%7Bi%7D) 라는 벡터로 정규화 된다        

​       

우선 __평균과 분산을 통해 벡터 ![image](https://latex.codecogs.com/gif.latex?x_%7Bi%7D) 를 정규화 해준다__        

![image](https://latex.codecogs.com/gif.latex?x_%7Bi%7D)는 벡터고 평균과 분산은 스칼리이기 때문에       

![image](https://latex.codecogs.com/gif.latex?x_%7Bi%7D) 의 각 차원 k 마다        

![image](https://latex.codecogs.com/gif.latex?%5Chat%7Bx%7D_%7Bi%2C%20k%7D%20%3D%20%5Cfrac%7Bx_%7Bi%2C%20k%7D-%5Cmu%20_%7Bi%7D%7D%7B%5Csqrt%7B%5Csigma%20%5E%7B2%7D_%7Bi%7D&plus;%5Cepsilon%7D%7D)    로 정규화 된다       

(입실론은 분모가 0이 되는 것을 방지)        

​       

다음 __감마와 베타 이용 (초기값 각각 1, 0  학습 가능한 파라미터)__       

![image](https://wikidocs.net/images/page/31379/%EA%B0%90%EB%A7%88%EB%B2%A0%ED%83%80.PNG)  

​          

층 정규화의 최종 수식        

![image](https://latex.codecogs.com/gif.latex?ln_%7Bi%7D%20%3D%20%5Cgamma%20%5Chat%7Bx%7D_%7Bi%7D&plus;%5Cbeta%20%3D%20LayerNorm%28x_%7Bi%7D%29)         



<br/>         



<br/>       

<br/>



### 디코더       

![image](https://wikidocs.net/images/page/31379/decoder.PNG)       

디코더도 인코더와 동일하게 임베딩 층과 포지셔널 인코딩을 거친 후의 문장 행렬이 입력됨      

seq2seq와 마찬가지로 __교사 강요(Teacher Forcing)__ 을 사용하여 훈련됨       

디코더는 이 문장 행렬로부터 각 시점의 단어를 예측하도록 훈련됨

​            

#### Masked Multi-head Self-Attention       

__Seq2Seq 의 디코더에서는 순차적으로 입력을 받아 다음 단어 예측에 이전에 입력된 단어들만 참고 가능!__     

__Transformer는 입력으로 한 번에 문장 행렬을 받았기 때문에 이후 시점의 단어까지도 예측에 참고할 수 있는 문제 발생!__      

​       

__이를 위해 디코더 첫 번째 서브층에서 현재 이후 시점에 있는 단어들을 참고하지 못하게 마스킹 하는 look-ahead mask 사용__        

_인코더의 첫 번째 서브층(Multi-head Self-Attention)은 Attention Score Matrix에 Padding Mask 사용_          

_디코더의 첫 번째 서브층(Masked Multi-head Self-Attention)에서  Attention Score Matrix에 Look-ahead mask 사용_           

_디코더의 두 번째 서브층(Encoder-Decoder Attention)에서  Attention Score Matrix에 Pading mask 사용_           

​       

![image](https://wikidocs.net/images/page/31379/decoder_attention_score_matrix.PNG)   

Self Attention을 통해 Attention Score Matrix 얻는다       

​       

![image](https://wikidocs.net/images/page/31379/%EB%A3%A9%EC%96%B4%ED%97%A4%EB%93%9C%EB%A7%88%EC%8A%A4%ED%81%AC.PNG) 

자기 자신과 이전 단어들만을 참고할 수 있게 마스킹 적용 

<br/>

#### Multi-head Attention(Encoder-Decoder Attention)        

디코더의 두 번째 서브층은 __Multi-head Attention을 수행하지만, Self Attention은 아니다__      

![image](https://wikidocs.net/images/page/31379/%EB%94%94%EC%BD%94%EB%8D%94%EB%91%90%EB%B2%88%EC%A7%B8%EC%84%9C%EB%B8%8C%EC%B8%B5.PNG) 

Query : Decoder 행렬    (검은색 화살표 : 디코더의 첫 번째 서브층의 결과 행렬)      

Key=Value : Encoder 행렬   (두 빨간색 화살표  :인코더의 마지막 층에서 온 행렬)        

![image](https://wikidocs.net/images/page/31379/%EB%94%94%EC%BD%94%EB%8D%94%EB%91%90%EB%B2%88%EC%A7%B8%EC%84%9C%EB%B8%8C%EC%B8%B5%EC%9D%98%EC%96%B4%ED%85%90%EC%85%98%EC%8A%A4%EC%BD%94%EC%96%B4%ED%96%89%EB%A0%AC_final.PNG) 

<br/>

<br/>

<br/>

<br/>

## Transformer 요약       

__인코더 n개 - 디코더 n개 구조__      

__Attention만으로 구현__      

​         

![image](https://wikidocs.net/images/page/31379/transformer_attention_overview.PNG)

### embedding + Positional Encoding?       

RNN 은 순차적으로 단어 입력 받음       

-> 각 단어에 위치 정보 내재      

Transformer는 문장 전체 한 번에 입력 받음      

-> 단어 임베딩 벡터에 위치 정보 더해줘야 함       

[Positional Encoding](#Positional-Encoding) 

이로써 Transformer의 입력은 __순서 정보가 고려된 임베딩 벡터__  됨         

<br/>

<br/>

![image](https://wikidocs.net/images/page/31379/transformer_attention_overview.PNG)

### 2가지 Self-Attention     

인코더의 Multi-head __Self-Attention__      

디코더의 Masked Multi-head __Self-Attention__                   

__(1) 문장 행렬에 가중치 행렬을 곱해 Q 행렬, K 행렬, V 행렬을 구한다__        

![image](https://wikidocs.net/images/page/31379/transformer12.PNG)



__(2) Q 벡터와 K 벡터를 내적한 결과 행렬 값에 전체적으로 ![image](https://latex.codecogs.com/gif.latex?%5Csqrt%7Bd_%7Bk%7D%7D) 를 나누어 Attention Score 행렬 구한다__          

![image](https://wikidocs.net/images/page/31379/transformer15.PNG)  

​         

__(3) Attention Score 행렬에 softmax  함수 적용해 Attention Distribution 행렬 구한다__       

__(4) 마지막으로 V 행렬 곱해  Attention Value 행렬 얻는다__     



![image](https://wikidocs.net/images/page/31379/transformer16.PNG)

 ![image](https://latex.codecogs.com/gif.latex?Attention%28Q%2C%20K%2C%20V%29%20%3D%20softmax%28%7BQK%5ET%5Cover%7B%5Csqrt%7Bd_k%7D%7D%7D%29V) 

​         

<br/>

### 두 Self-Attention의 차이      

__Padding Mask vs Look-Ahead Mask__       

인코더의 Multi-head Self-Attention은 Padding Mask 적용         

![image](https://wikidocs.net/images/page/31379/pad_masking2.PNG)

단어 간 유사도를 구하는 일에 실질적인 의미를 가지지 않은 '<패드>' 에 대해 유사도를 구하지 않도록       

이후 Softmax 함수를 지나면서 0에 굉장히 가까운 값이 됨         

​         

디코더의 Masked Multi-head Self-Attention은 Padding Mask, Look-Ahead Mask 둘 다 적용       

![image](https://wikidocs.net/images/page/31379/%EB%A3%A9%EC%96%B4%ED%97%A4%EB%93%9C%EB%A7%88%EC%8A%A4%ED%81%AC.PNG)

자기 자신과 이전 단어들만을 참고할 수 있게 마스킹 적용 

​      

<br/>

![image](https://wikidocs.net/images/page/31379/transformer_attention_overview.PNG)

### Multi-head Attention(Encoder-Decoder Attention)      

![image](https://wikidocs.net/images/page/31379/%EB%94%94%EC%BD%94%EB%8D%94%EB%91%90%EB%B2%88%EC%A7%B8%EC%84%9C%EB%B8%8C%EC%B8%B5.PNG)

![image](https://wikidocs.net/images/page/31379/%EB%94%94%EC%BD%94%EB%8D%94%EB%91%90%EB%B2%88%EC%A7%B8%EC%84%9C%EB%B8%8C%EC%B8%B5%EC%9D%98%EC%96%B4%ED%85%90%EC%85%98%EC%8A%A4%EC%BD%94%EC%96%B4%ED%96%89%EB%A0%AC_final.PNG)

Query : Decoder 행렬    (검은색 화살표 : 디코더의 첫 번째 서브층의 결과 행렬)       

Key=Value : Encoder 행렬   (두 빨간색 화살표  :인코더의 마지막 층에서 온 행렬)        

​            

<br/>

 ![image](https://wikidocs.net/images/page/31379/transformer_attention_overview.PNG)

### 3가지 Multi-head Attention            

인코더의 __Multi-head__  Self-Attention         

디코더의 Masked __Multi-head__  Self-Attention       

디코더의 __Multi-head__  Attention(Encoder-Decoder Attention)

​        ![image](https://wikidocs.net/images/page/31379/transformer17.PNG)

![image](https://wikidocs.net/images/page/31379/transformer18_final.PNG)     



![image](https://wikidocs.net/images/page/31379/transformer19.PNG)

__인코더의 입력이었던 문장 행렬의 크기 ![image](https://latex.codecogs.com/gif.latex?%28seq%5C%3A%20%5C%3A%20len%2C%5C%20d_%7Bmodel%7D%29)   와 동일__      



Transformer는 다수의 인코더를 쌓은 형태이기 때문에 __인코더에서의 입력의 크기가 출력에서도 동일 크기로 계속 유지되어야만__  다음 인코더에서도 다시 입력이 될 수 있다       

​         

<br/>

<br/>

![image](https://wikidocs.net/images/page/31379/transformer_attention_overview.PNG)

### Position-wise FFNN        

![image](https://latex.codecogs.com/gif.latex?FFNN%28x%29%20%3D%20MAX%280%2C%20x%7BW_%7B1%7D%7D%20&plus;%20b_%7B1%7D%29%7BW_2%7D%20&plus;%20b_2)

여기서 x는 Multi-head attention 결과 나온 ![image](https://latex.codecogs.com/gif.latex?%28seq%5C%3A%20%5C%3A%20len%2C%5C%20d_%7Bmodel%7D%29) 크기 행렬        

```python
outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
outputs = tf.keras.layers.Dense(units=d_model)(outputs)
```

![image](https://wikidocs.net/images/page/31379/positionwiseffnn.PNG)

__출력은 입력의 크기 였던 ![image](https://latex.codecogs.com/gif.latex?%28seq%5C%3A%20%5C%3A%20len%2C%5C%20d_%7Bmodel%7D%29) 의 크기 와 동일__        

​         

<br/>

### Add & Norm       

__잔차 연결(residual connection)__ 과 __층 정규화(layer normalization)__         

__잔차 연결__       

![image](https://wikidocs.net/images/page/31379/transformer22.PNG)

​         

![image](https://latex.codecogs.com/gif.latex?x&plus;Sublayer%28x%29)



__층 정규화__       

![image](https://latex.codecogs.com/gif.latex?LN%20%3D%20LayerNorm%28x&plus;Sublayer%28x%29%29)        

잔차 연결을 거친 결과는 이어서 층 정규화 거친다  

![image](https://wikidocs.net/images/page/31379/layer_norm_new_2_final.PNG)

​       

![image](https://latex.codecogs.com/gif.latex?%5Chat%7Bx%7D_%7Bi%2C%20k%7D%20%3D%20%5Cfrac%7Bx_%7Bi%2C%20k%7D-%5Cmu%20_%7Bi%7D%7D%7B%5Csqrt%7B%5Csigma%20%5E%7B2%7D_%7Bi%7D&plus;%5Cepsilon%7D%7D)

​          

다음 __감마와 베타 이용 (초기값 각각 1, 0  학습 가능한 파라미터)__       

![image](https://wikidocs.net/images/page/31379/%EA%B0%90%EB%A7%88%EB%B2%A0%ED%83%80.PNG)  

​          

층 정규화의 최종 수식        

![image](https://latex.codecogs.com/gif.latex?ln_%7Bi%7D%20%3D%20%5Cgamma%20%5Chat%7Bx%7D_%7Bi%7D&plus;%5Cbeta%20%3D%20LayerNorm%28x_%7Bi%7D%29)            





