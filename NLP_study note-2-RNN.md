# 딥 러닝을 이용한 자연어 처리 입문     

​     

RNN(Recurrent Neural Network)은 딥 러닝에 있어 가장 기본적인 __시퀀스(Sequence)__ 모델 즉, 입력과 출력을 시퀀스 단위로 처리하는 모델이다       

<br/>

## 1. 순환 신경망 (Recurrent Neural Network, RNN)     

RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보낸다     

![image](https://wikidocs.net/images/page/22886/rnn_image1_ver2.PNG)

_편향 b는 그림에서 생략_        

​     

RNN에서는 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드를 __셀(cell)__       

이 셀은 이전의 값을 기억하려는 일종의 메모리 역할을 수행하므로 **메모리 셀** 또는 **RNN 셀**라고 표현      

<br/>

![image](https://wikidocs.net/images/page/22886/rnn_image2_ver3.PNG)

t  :  현재 시점     

__은닉상태 (hidden state)__  :  메모리 셀이 출력층 방향으로 또는 다음 시점 t+1의 자신에게 보내는 값           

t 시점의 메모리 셀은 t-1 시점의 메모리 셀이 보낸 은닉 상태값을 은닉 상태 계산을 위한 입력값으로 사용      

<br/>

<br/>

![image](https://wikidocs.net/images/page/22886/rnn_image3_ver2.PNG)

RNN은 입력과 출력의 길이를 다르게 설계 할 수 있으므로 다양한 용도로 사용할 수 있다     

![image](https://wikidocs.net/images/page/22886/rnn_image3.5.PNG)                            ![image](https://wikidocs.net/images/page/22886/rnn_image3.7.PNG)

<br/>

<br/>

![image](https://wikidocs.net/images/page/22886/rnn_image4_ver2.PNG)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20h_%7Bt%7D)  : 현재 시점 t 에서의 은닉 상태 값      

은닉층  :  ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20h_%7Bt%7D%20%3D%20tanh%28W_%7Bx%7D%20x_%7Bt%7D%20&plus;%20W_%7Bh%7D%20h_%7Bt-1%7D%20&plus;%20b%29)

출력층  :  ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20y_%7Bt%7D%20%3D%20f%28W_%7By%7Dh_%7Bt%7D%20&plus;%20b%29)

_f는 비선형 활성화 함수 중 하나_      

​      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20d) : 단어 벡터의 차원      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20D_%7Bh%7D)  :  은닉 상태의 크기      

Wx  (은닉 상태의 크기 × 입력의 차원)      

Wh  (은닉 상태의 크기 × 은닉 상태의 크기)       

b     (은닉 상태의 크기)

![image](https://wikidocs.net/images/page/22886/rnn_images4-5.PNG)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20h_%7Bt%7D)를 계산하기 위한 활성화 함수로 주로 tanh 사용되지만 ReLU로 사용하는 시도도 있다      

__만약, 은닉층이 2개 이상일 경우에는 은닉층 2개의 가중치는 서로 다르다__     

<br/>

<br/>

![image](https://wikidocs.net/images/page/22886/rnn_image7_ver2.PNG)         

<br/>

RNN 층은 사용자의 설정에 따라 두 가지 종류의 출력을 내보낸다     

메모리 셀의 최종 시점의 은닉 상태만을 리턴하고자 한다면 (batch_size, output_dim) 크기의 2D 텐서를 리턴       

메모리 셀의 각 시점(time step)의 은닉 상태값들을 모아서 전체 시퀀스를 리턴하고자 한다면 (batch_size, timesteps, output_dim) 크기의 3D 텐서를 리턴      

__RNN 층의 return_sequences 매개 변수에 True, False를 설정하여 설정__

![image](https://wikidocs.net/images/page/22886/rnn_image8_ver2.PNG)

마지막 은닉 상태만 전달하도록 하면 many-to-one 문제를 풀 수 있고,      

모든 시점의 은닉 상태를 전달하도록 하면, 다음층에 은닉층이 하나 더 있는 경우이거나 many-to-many 문제를 풀 수 있다     

```python
model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8,2,10)))
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_2 (SimpleRNN)     (8, 3)                    42        
=================================================================
Total params: 42
Trainable params: 42
Non-trainable params: 0
_________________________________________________________________
```

```python
model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8,2,10), return_sequences=True))
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_3 (SimpleRNN)    (8, 2, 3)                 42        
=================================================================
Total params: 42
Trainable params: 42
Non-trainable params: 0
_________________________________________________________________
```

<br/>

--------------

<br/>

## 2. 깊은 순환 신경망 (Deep Recurrent Neural Network)    

![image](https://wikidocs.net/images/page/22886/rnn_image4.5_finalPNG.PNG)

```python
model = Sequential()
model.add(SimpleRNN(hidden_size, return_sequences = True))
model.add(SimpleRNN(hidden_size, return_sequences = True))
```

<br/>

-----------

<br/>

## 3. 양방향 순환 신경망 (Bidirectional Recurrent Neural Network)     

실제 문제에서는 과거 시점의 데이터만 향후 시점의 데이터에도 힌트가 있는 경우도 많다     

그래서 이전 시점의 데이터뿐만 아니라, 이후 시점의 데이터도 힌트로 활용하기 위해서 고안된 것이 __양방향 RNN__      

<br/>

![image](https://wikidocs.net/images/page/22886/rnn_image5_ver2.PNG)

하나의 출력값을 예측하기 위해 기본적으로 __두 개의 메모리 셀__을 사용     

첫번째 메모리 셀 (주황색 메모리 셀)  :  이전처럼 **앞 시점의 은닉 상태(Forward States)**를 전달받아 현재의 은닉 상태 계산      

두번째 메모리 셀 (초록색 메모리 셀)  :  **뒤 시점의 은닉 상태(Backward States)**를 전달 받아 현재의 은닉 상태를 계산     

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Bidirectional

model = Sequential()
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences = True), input_shape=(timesteps, input_dim)))
```

<br/>

양방향 RNN도 다수의 은닉층을 가질 수 있다      

![image](https://wikidocs.net/images/page/22886/rnn_image6_ver3.PNG)

```python
model = Sequential()
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences = True), input_shape=(timesteps, input_dim)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences = True)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences = True)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences = True)))
```

<br/>

<br/>



----------------------

<br/>

## Keras 로 실습     

임의의 입력 생성 

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, LSTM, Bidirectional

train_X = [[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]
print(np.shape(train_X))
(4, 5) 
# 단어 벡터 차원        5  
# 문장 길이(timesteps) 4
```

__RNN은 2D 텐서가 아니라 3D 텐서를 입력을 받는다!__       

__2D 텐서를 배치 크기 1을 추가해줌으로써 3D 텐서로 변경__     



```python
train_X = [[[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]]
train_X = np.array(train_X, dtype=np.float32)
print(train_X.shape)
(1, 4, 5)
# (batch_size, timesteps, input_dim) 가 (1, 4, 5)의 크기를 가지는 3D 텐서
```

​     

SimpleRNN에 대표적인 parameter로 __return_sequences__ 와 __return_state__ 가 있다     

기본값으로는 둘 다 False     

```python
rnn = SimpleRNN(3)
# rnn = SimpleRNN(3, return_sequences=False, return_state=False)와 동일.
hidden_state = rnn(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))

hidden state : [[-0.866719    0.95010996 -0.99262357]], shape: (1, 3)
```

(1, 3) 크기의 텐서 출력  <-  __마지막 시점의 은닉 상태__     

기본적으로 __return_sequences__가 __False__인 경우에 __SimpleRNN은 마지막 시점의 은닉 상태만 출력__     



__return_sequences__를 __True__로 지정해 모든 시점의 은닉 상태를 출력

```python
rnn = SimpleRNN(3, return_sequences=True)
hidden_states = rnn(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))

hidden states : [[[ 0.92948604 -0.9985648   0.98355013]
  [ 0.89172053 -0.9984244   0.191779  ]
  [ 0.6681082  -0.96070355  0.6493537 ]
  [ 0.95280755 -0.98054564  0.7224146 ]]], shape: (1, 4, 3)
```

(1, 4, 3) 크기의 텐서 출력  <- __모든 시점(timesteps)에 대해 은닉 상태 값 출력해 (1, 4, 3) 크기의 텐서 출력__      

​     

__return_state__가 __True__일 경우 __return_sequences__의 __True/False__ 여부와 상관없이 마지막 시점의 은닉 상태 출력       

__return_sequences__가 __True__이면서, __return_state__를 __True__로 할 경우 SimpleRNN은 두 개의 출력 리턴

```python
rnn = SimpleRNN(3, return_sequences=True, return_state=True)
hidden_states, last_state = rnn(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))

hidden states : [[[ 0.29839835 -0.99608386  0.2994854 ]
  [ 0.9160876   0.01154806  0.86181474]
  [-0.20252597 -0.9270214   0.9696659 ]
  [-0.5144398  -0.5037417   0.96605766]]], shape: (1, 4, 3)
last hidden state : [[-0.5144398  -0.5037417   0.96605766]], shape: (1, 3)
```

__첫번째 출력__  <-  __return_sequences=True__로 __모든 시점의 은닉 상태__ 출력

__두번째 출력__  <-  __return_state=True__로 __마지막 시점의 은닉 상태__ 출력     

​     

__return_sequences__는 __False__인데, __retun_state__가 __True__인 경우

```python
rnn = SimpleRNN(3, return_sequences=False, return_state=True)
hidden_state, last_state = rnn(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))
hidden state : [[0.07532981 0.97772664 0.97351676]], shape: (1, 3)
last hidden state : [[0.07532981 0.97772664 0.97351676]], shape: (1, 3)
```

__두 개의 출력 모두 마지막 시점의 은닉 상태 출력__ 