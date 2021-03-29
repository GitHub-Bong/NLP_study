# 복습     

#      

# LSTM, Long Short-Term Memory      

<br/>

## 1. RNN의 한계      


__장기 의존성 문제__ (problem of Long-Term Dependencies)     

RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생     

![image](https://wikidocs.net/images/page/22888/lstm_image1_ver2.PNG)

뒤로 갈수록 x1의 정보량은 손실되고, 시점이 충분히 긴 상황에서는 x1의 전체 정보에 대한 영향력은 거의 의미가 없을 수도 있다      

<br/>

## 2. RNN 내부       

![image](https://wikidocs.net/images/page/22888/vanilla_rnn_ver2.PNG)

편향 b 생략     

편향 b를 그린다면 xt 옆에 tanh로 향하는 또 하나의 입력선을 그리면 된다     

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20h_%7Bt%7D%20%3D%20tanh%28W_%7Bx%7Dx_%7Bt%7D%20&plus;%20W_%7Bh%7Dh_%7Bt-1%7D%20&plus;%20b%29)

 xt와 ht−1 두 개의 입력이 각각의 가중치와 곱해져서 메모리 셀의 입력이 된다     

이를 tanh 함수의 입력으로 사용하고 이 값은 은닉층의 출력인 은닉 상태가 된다!      

<br/>

<br/>

## 3. LSTM (Long Short-Term Memory)      

![image](https://wikidocs.net/images/page/22888/vaniila_rnn_and_different_lstm_ver2.PNG)

LSTM은 은닉층의 메모리 셀에 __입력 게이트__, __삭제 게이트__, __출력 게이트__ 를 추가하여 불필요한 기억을 지우고, 기억해야할 것들을 정한다      

은닉 상태(hidden state)를 계산하는 식이 전통적인 RNN보다 조금 더 복잡해졌으며 __셀 상태(cell state)__ 라는 값이 추가됐다        

​      

LSTM은 RNN과 비교하여 긴 시퀀스의 입력을 처리하는데 탁월한 성능을 보인다     

<br/>

![image](https://wikidocs.net/images/page/22888/cellstate.PNG)

셀 상태 : 왼쪽에서 오른쪽으로 가는 굵은 선

이전 시점의 셀 상태가 다음 시점의 셀 상태를 구하기 위한 입력으로서 사용된다     

<br/>

삭제 게이트, 입력 게이트, 출력 게이트 이 3개의 게이트에는 공통적으로 sigmoid 함수가 존재  

<br/>

__(1) 입력 게이트__       

![image](https://wikidocs.net/images/page/22888/inputgate.PNG)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20i_%7Bt%7D%3D%5Csigma%20%28W_%7Bxi%7Dx_%7Bt%7D&plus;W_%7Bhi%7Dh_%7Bt-1%7D&plus;b_%7Bi%7D%29)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20g_%7Bt%7D%3Dtanh%28W_%7Bxg%7Dx_%7Bt%7D&plus;W_%7Bhg%7Dh_%7Bt-1%7D&plus;b_%7Bg%7D%29)

입력 게이트는 __현재 정보를 기억하기 위한 게이트__       

sigmoid 함수를 지나 0과 1 사이의 값과 tanh 함수를 지나 -1과 1사이의 값 두 개가 나오게 된다      

이 두 개의 값을 가지고 이번에 선택된 기억할 정보의 양을 정한다      

<br/>

__(2) 삭제 게이트__     

![image](https://wikidocs.net/images/page/22888/forgetgate.PNG)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20f_%7Bt%7D%3D%5Csigma%20%28W_%7Bxf%7Dx_%7Bt%7D&plus;W_%7Bhf%7Dh_%7Bt-1%7D&plus;b_%7Bf%7D%29)

삭제 게이트는 __기억을 삭제하기 위한 게이트__     

sigmoid 함수를 지나면 0과 1 사이의 값이 나오게 되는데, 이 값이 곧 __삭제 과정을 거친 정보의 양__      

0에 가까울수록 정보가 많이 삭제된 것이고 1에 가까울수록 정보를 온전히 기억한 것     

<br/>

__(3) 셀 상태(장기 상태)__      

![image](https://wikidocs.net/images/page/22888/cellstate2.PNG)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20C_%7Bt%7D%3Df_%7Bt%7D%5Ccirc%20C_%7Bt-1%7D&plus;i_%7Bt%7D%5Ccirc%20g_%7Bt%7D)

입력 게이트에서 구한 it, gt 두 개의 값에 대해서 원소별 곱(entrywise product)을 진행     

(같은 크기의 두 행렬이 있을 때 같은 위치의 성분끼리 곱)      

__이것이 이번에 선택된 기억할 값__

입력 게이트에서 선택된 기억을 삭제 게이트의 결과값과 더한다     

이 값이 __현재 시점 t의 셀 상태__          



만약 삭제 게이트의 출력값 ft가 0이 된다면,       

-> 이전 시점의 셀 상태값인 Ct−1은 현재 시점의 셀 상태값을 결정하기 위한 영향력이 0     

-> 오직 입력 게이트의 결과만이 현재 시점의 셀 상태값 Ct을 결정       

--> 삭제 게이트가 완전히 닫히고 입력 게이트를 연 상태를 의미     

​       

입력 게이트의 it값을 0이라고 한다면,      

-> 현재 시점의 셀 상태값 Ct는 오직 이전 시점의 셀 상태값 Ct−1값에만 의존

-> 이는 입력 게이트를 완전히 닫고 삭제 게이트만을 연 상태를 의미       

​         

__결과적으로 삭제 게이트는 이전 시점의 입력을 얼마나 반영할지를 의미하고, 입력 게이트는 현재 시점의 입력을 얼마나 반영할지를 결정__

<br/>

__(4) 출력 게이트와 은닉 상태(단기 상태)

![image](https://wikidocs.net/images/page/22888/outputgateandhiddenstate.PNG)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20o_%7Bt%7D%3D%5Csigma%20%28W_%7Bxo%7Dx_%7Bt%7D&plus;W_%7Bho%7Dh_%7Bt-1%7D&plus;b_%7Bo%7D%29)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20h_%7Bt%7D%3Do_%7Bt%7D%5Ccirc%20tanh%28c_%7Bt%7D%29)

출력 게이트는 현재 시점 t의 x값과 이전 시점 t-1의 은닉 상태가 sigmoid 함수를 지난 값     

장기 상태의 값이 tanh함수를 지나 -1과 1사이의 값이 되고, 출력 게이트의 값과 연산되면서,     

값이 걸러지는 효과가 발생하여 은닉 상태가 된다! 

<br/>

<br/>

------

<br/>

## Keras 로 실습     

​     

__return_sequences__ 가 __False__, __return_state__ 가 __True__ 인 경우     

```python
lstm = LSTM(3, return_sequences=False, return_state=True)
hidden_state, last_state, last_cell_state = lstm(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))
print('last cell state : {}, shape: {}'.format(last_cell_state, last_cell_state.shape))
hidden state : [[-0.00263056  0.20051427 -0.22501363]], shape: (1, 3)
last hidden state : [[-0.00263056  0.20051427 -0.22501363]], shape: (1, 3)
last cell state : [[-0.04346419  0.44769213 -0.2644241 ]], shape: (1, 3)
```

__SimpleRNN 때와는 달리, 세 개의 결과 반환__     

__첫 번째 결과__  <-  __return_sequences__ 가 __False__ 이므로 __마지막 시점의 은닉 상태__ 출력     

​     

__SimpleRNN과 다른 점!__     

__return_state__ 를 __True__ 로 둔 경우 마지막 시점의 은닉 상태뿐만 아니라 __셀 상태까지 반환__      

​     

__return_sequences__ 가 __True__ 인 경우

```python
lstm = LSTM(3, return_sequences=True, return_state=True)
hidden_states, last_hidden_state, last_cell_state = lstm(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('last hidden state : {}, shape: {}'.format(last_hidden_state, last_hidden_state.shape))
print('last cell state : {}, shape: {}'.format(last_cell_state, last_cell_state.shape))
hidden states : [[[ 0.1383949   0.01107763 -0.00315794]
  [ 0.0859854   0.03685492 -0.01836833]
  [-0.02512104  0.12305924 -0.0891041 ]
  [-0.27381724  0.05733536 -0.04240693]]], shape: (1, 4, 3)
last hidden state : [[-0.27381724  0.05733536 -0.04240693]], shape: (1, 3)
last cell state : [[-0.39230722  1.5474017  -0.6344505 ]], shape: (1, 3)
```

__return_sequences__ 가 __True__ 이므로 첫번째 출력값은 __모든 시점의 은닉 상태__ 출력

<br/>

<br/>

## Bidirectional (LSTM)   이해하기     

​     

출력되는 은닉 상태 값 고정

```python
k_init = tf.keras.initializers.Constant(value=0.1)
b_init = tf.keras.initializers.Constant(value=0)
r_init = tf.keras.initializers.Constant(value=0.1)
```

​     

__return_sequences__ 가 __False__ 이고, __return_state__ 가 __True__ 인 경우

```python
bilstm = Bidirectional(LSTM(3, return_sequences=False, return_state=True, \
                            kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(train_X)


print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('forward state : {}, shape: {}'.format(forward_h, forward_h.shape))
print('backward state : {}, shape: {}'.format(backward_h, backward_h.shape))

hidden states : [[0.6303139  0.6303139  0.6303139  0.70387346 0.70387346 0.70387346]], shape: (1, 6)
forward state : [[0.6303139 0.6303139 0.6303139]], shape: (1, 3)
backward state : [[0.70387346 0.70387346 0.70387346]], shape: (1, 3)
```

![image](https://wikidocs.net/images/page/94748/bilstm3.PNG)

__5__ 개의 값 반환      

__return_state__ 가 __True__ 인 경우 정방향 LSTM의 은닉 상태와 셀 상태, 역방향 LSTM의 은닉 상태와 셀 상태 4가지를 반환      

__forward_h__ 와 __backward_h__ 는 각각 정방향 LSTM의 마지막 시점의 은닉 상태와 역방향 LSTM의 첫번째 시점의 은닉 상태값     

​	-> __이 두 값을 연결한 값이 hidden_states에 출력되는 값__

​       

__return_sequences__  가 __False__ 인 경우 정방향 LSTM의 마지막 시점의 은닉 상태와 역방향 LSTM의 첫번째 시점의 은닉 상태가 연결된 채 반환 

​	-> __첫번째 출력값의 크기 (1, 6)__      



셀 상태는 각각 forward_c와 backward_c에 저장만 하고 출력하지 않았다      

<br/>

현재 은닉 상태의 값을 고정시켜두었기 때문에 __return_sequences__ 를 __True__ 로 할 경우, 출력이 어떻게 바뀌는지 비교가 가능     

```python
bilstm = Bidirectional(LSTM(3, return_sequences=True, return_state=True, \
                            kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(train_X)


print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('forward state : {}, shape: {}'.format(forward_h, forward_h.shape))
print('backward state : {}, shape: {}'.format(backward_h, backward_h.shape))

hidden states : [[[0.3590648  0.3590648  0.3590648  0.70387346 0.70387346 0.70387346]
  [0.5511133  0.5511133  0.5511133  0.5886358  0.5886358  0.5886358 ]
  [0.5911575  0.5911575  0.5911575  0.39516988 0.39516988 0.39516988]
  [0.6303139  0.6303139  0.6303139  0.21942243 0.21942243 0.21942243]]], shape: (1, 4, 6)
forward state : [[0.6303139 0.6303139 0.6303139]], shape: (1, 3)
backward state : [[0.70387346 0.70387346 0.70387346]], shape: (1, 3)
```

![image](https://wikidocs.net/images/page/94748/bilstm1.PNG)

__hidden states__ 의 출력값에서 이제 __모든 시점의 은닉 상태 출력__     

__역방향 LSTM의 첫번째 시점의 은닉 상태__ 는 더 이상 __정방향 LSTM의 마지막 시점의 은닉 상태__ 와 연결되는 것이 아니라 __정방향 LSTM의 첫번째 시점의 은닉 상태__ 와 연결된다    

