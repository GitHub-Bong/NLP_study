# 딥 러닝을 이용한 자연어 처리 입문

<br/>

신경망 언어 모델의 시초인 피드 포워드 신경망 언어 모델(Feed Forward Neural Network Language Model) 부터 시작한다.

<br/>

# Neural Network Language Model, NNLM     

<br/>

## 1. 기존 N-gram 언어 모델의 한계

![image](https://wikidocs.net/images/page/21692/n-gram.PNG)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Ctiny%20P%28w%5Ctext%7B%7Cboy%20is%20spreading%7D%29%20%3D%20%5Cfrac%7B%5Ctext%7Bcount%28boy%20is%20spreading%7D%5C%20w%29%7D%7B%5Ctext%7Bcount%28boy%20is%20spreading%29%7D%7D)

n-gram 언어 모델은 충분한 데이터를 관측하지 못하면 언어를 정확히 모델링하지 못하는 **희소 문제(sparsity problem)**가 있다!      

​     

__희소 문제는 기계가 단어 간 유사도를 알 수 있다면 해결할 수 있는 문제!__      

__훈련 코퍼스에 없는 단어 시퀀스에 대한 예측이라도 유사한 단어가 사용된 단어 시퀀스를 참고하여 보다 정확한 예측을 할 수 있을것이다!__      

-> __NNLM__ , __word embedding__의 아이디어!     

<br/>

------------

<br/>

## 2. NNLM     

Example)     

**예문 : "what will the fat cat sit on"**      

언어 모델은 주어진 단어 시퀀스로부터 다음 단어를 예측하는 모델      

'what will the fat cat'이라는 단어 시퀀스가 입력으로 주어지면, 다음 단어 'sit'을 예측하는 방식으로 훈련!       

​      

단어들에 대해서 __원-핫 인코딩__     

```python
what = [1, 0, 0, 0, 0, 0, 0]
will = [0, 1, 0, 0, 0, 0, 0]
the = [0, 0, 1, 0, 0, 0, 0]
fat = [0, 0, 0, 1, 0, 0, 0]
cat = [0, 0, 0, 0, 1, 0, 0]
sit = [0, 0, 0, 0, 0, 1, 0]
on = [0, 0, 0, 0, 0, 0, 1]
```

'what will the fat cat'를 입력 받아 'sit'을 예측하는 일       

 -> 기계에게 what, will, the, fat, cat의 원-핫 벡터를 입력받아 sit의 원-핫 벡터를 예측하는 문제      

​      

NNLM은 앞의 모든 단어를 참고하는 것이 아니라 정해진 n개의 단어만을 참고 (n-gram과 유사)       

__윈도우(window)__ : 참고하는 범위                      ex) 앞 4개 단어 참고 n=4 , 윈도우의 크기 = 4     

<br/>__NNLM의 구조__     

총 4개의 층(layer)으로 이루어진 인공 신경망

![image](https://wikidocs.net/images/page/45609/nnlm1.PNG)

__입력층(input layer)__  : 입력은 4개의 단어 'will, the, fat, cat'의 원-핫 벡터     

__출력층(Output layer)__  : 정답 단어 sit의 원-핫 벡터는 출력층에서 모델의 예측값의 오차를 구하기 위해 사용될 예정      

이 오차로부터 손실 함수를 사용해 인공 신경망이 학습     

​      

4개의 원-핫 벡터를 입력 받은 NNLM은 다음층인 __투사층(projection layer)__을 지나게된다!      

투사층이 일반 은닉층과 구별되는 특징은 __가중치 행렬과의 연산은 이루어지지만 활성화 함수가 존재하지 않는다는 것__       

​      

![image](https://wikidocs.net/images/page/45609/nnlm2_renew.PNG)

단어 집합의 크기 : V        투사층의 크기 : M          

-> 투사층의 가중치 행렬의 크기  : V x M

​     

i번째 인덱스에 1이라는 값을 가지고 그 외의 0의 값을 가지는 원-핫 벡터와 가중치 W 행렬의 곱     

__사실 W행렬의 i번째 행을 그대로 읽어오는 것과(lookup) 동일!__     

이 작업을 __룩업 테이블 (lookup table)__      

​      

V 차원의 **원-핫 벡터**  ->  __lookup table 작업__  ->  M 차원의 단어 벡터       

이 벡터들은 초기에는 랜덤한 값을 가지지만 학습 과정에서 값이 계속 변경된다!      

이 단어 벡터를 **임베딩 벡터(embedding vector)**      

<br/>

![image](https://wikidocs.net/images/page/45609/nnlm3_renew.PNG)

각 단어가 __테이블 룩업__을 통해 임베딩 벡터로 변경  ->    

투사층에서 모든 임베딩 벡터들의 값은 __연결(concatenation)__된다     

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Ctiny%20p%5E%7Blayer%7D%20%3D%20%28lookup%28x_%7Bt-n%7D%29%3B%20...%3B%20lookup%28x_%7Bt-2%7D%29%3B%20lookup%28x_%7Bt-1%7D%29%29%20%3D%20%28e_%7Bt-n%7D%3B%20...%3B%20e_%7Bt-2%7D%3B%20e_%7Bt-1%7D%29)

x : 각 단어의 원-핫 벡터       n : 윈도우의 크기       t : 예측하고자 하는 단어가 문장에서 t번째     

​     

__투사층은 활성화 함수가 존재하지 않는 선형층(linear layer)__

<br/>

![image](https://wikidocs.net/images/page/45609/nnlm4.PNG)

투사층의 결과는 h의 크기를 가지는 은닉층을 지난다     

-> 투사층의 결가는 __가중치와 곱해진 후 편향이 더해져 활성화 함수의 입력__이 된다     

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20h%5E%7Blayer%7D%20%3D%20tanh%28W_%7Bh%7Dp%5E%7Blayer%7D%20&plus;%20b_%7Bh%7D%29)     

<br/>

![image](https://wikidocs.net/images/page/45609/nnlm5_final.PNG)

은닉층의 출력은 이제 V의 크기를 가지는 출력층으로  향한다     

-> 또 다른 가중치와 곱해지고 편향이 더해지면, 입력이었던 원-핫 벡터와 동일한 V차원의 벡터!     

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Chat%7By%7D%20%3D%20softmax%28W_%7By%7Dh%5E%7Blayer%7D%20&plus;%20b_%7By%7D%29)

출력층의 활성화 함수로 Softmax function  사용      

NNLM은 손실함수로  Cross-Entropy function 사용

<br/>

<br/>

### NNLM의 핵심        

충분한 양의 훈련 코퍼스를 위와 같은 과정으로 학습한다면      

결과적으로 수많은 문장에서 유사한 목적으로 사용되는 단어들은 결국 유사한 임베딩 벡터값을 얻게 된다            

이렇게 되면 훈련이 끝난 후 다음 단어를 예측하는 과정에서 훈련 코퍼스에서 없던 단어 시퀀스라고 하더라도 다음 단어를 선택할 수 있다!        

​       



단어 간 유사도를 구할 수 있는 임베딩 벡터의 아이디어는 Word2Vec, FastText, GloVe 등으로 발전되어서 딥 러닝 모델에서는 필수적으로 사용되는 방법이 되었다        

<br/>

### NNLM의 이점과 한계     

밀집 벡터(dense vector)를 사용하므로서 단어의 유사도를 표현할 수 있다      

-> 희소 문제(sparsity problem) 해결!      

더 이상 모든 n-gram을 저장하지 않아도 된다      

-> n-gram 언어 모델보다 저장 공간의 이점! 

<br/>

__고정된 길이의 입력(Fixed-length-input)__     

정해진 n개의 단어만을 참고      

->  버려지는 단어들이 가진 문맥 정보는 참고할 수 없음!       

훈련 코퍼스에 각 문장의 길이는 전부 다를 수 있음      

-> 모델이 매번 다른 길이의 입력 시퀀스에 대해서도 처리할 수 있는 능력 필요!