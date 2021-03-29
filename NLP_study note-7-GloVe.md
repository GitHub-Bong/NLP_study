# 딥 러닝을 이용한 자연어 처리 입문     

​     

# GloVe      

GloVe(Global Vectors for Word Representation, GloVe)는 __카운트 기반__ 과 __예측 기반__ 을 모두 사용하는 단어 임베딩 방법론      

기존의 카운트 기반의 LSA(Latent Semantic Analysis)와 예측 기반의 Word2Vec의 단점을 이를 보완한다는 목적      

현재까지 연구에 따르면 단정적으로 Word2Vec와 GloVe 중에서 어떤 것이 더 뛰어나다고 말할 수는 없다      

<br/>

LSA는      

DTM이나 TF-IDF 행렬과 같이 각 문서에서의 __각 단어의 빈도수를 카운트 한 행렬__ 이라는 전체적인 통계 정보를 입력으로 받아 차원을 축소(Truncated SVD)하여 잠재된 의미를 끌어내는 방법론       

카운트 기반으로 코퍼스의 전체적인 통계 정보를 고려하기는 하지만      

단어 의미의 유추 작업(Analogy task)에는 성능이 떨어진다 

<br/>

Word2Vec는 실제값과 예측값에 대한 오차를 손실 함수를 통해 줄여나가며 __학습하는 예측 기반__의 방법론      

예측 기반으로 단어 간 유추 작업에는 LSA보다 뛰어나지만, 임베딩 벡터가 윈도우 크기 내에서만 주변 단어를 고려하기 때문에 코퍼스의 전체적인 통계 정보를 반영하지 못한다

<br/>

__GloVe는 LSA의 메커니즘이었던 카운트 기반의 방법과 Word2Vec의 메커니즘이었던 예측 기반의 방법론 두 가지를 모두 사용__     

<br/>

<br/>

-------

### 2. 윈도우 기반 동시 등장 행렬 (Window based Co-occurrence Matrix)       

단어의 동시 등장 행렬은 행과 열을 전체 단어 집합의 단어들로 구성하고,       

i 단어의 윈도우 크기(Window Size) 내에서 k 단어가 등장한 횟수를 i행 k열에 기재한 행렬      

Example)      

I like deep learning    

I like NLP   

I enjoy flying        

윈도우 크기가 1일 때 

<img src = "/image/GloVe 1.PNG" width = "500px">

<br/>

### 3. 동시 등장 확률 (Co-occurrence Probability)      

동시 등장 확률 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20P%28k%5C%20%7C%5C%20i%29)는 동시 등장 행렬로부터 특정 단어 i의 전체 등장 횟수를 카운트하고,        

특정 단어 i가 등장했을 때 어떤 단어 k가 등장한 횟수를 카운트하여 계산한 조건부 확률     

​     

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20P%28k%5C%20%7C%5C%20i%29) 에서  i를 중심 단어(Center Word), k를 주변 단어(Context Word)라고 했을 때,      

동시 등장 행렬에서 중심 단어 i의 행의 모든 값을 더한 값을 분모로 하고 i행 k열의 값을 분자로 한 값이라고 할 수 있다       

<img src = "/image/GloVe 2.PNG" width = "600px">

<img src = "/image/GloVe 3.PNG" width = "600px">

<br/>

--------

<br/>

### 4. Loss function     

GloVe의 아이디어     

**'임베딩 된 중심 단어와 주변 단어 벡터의 내적이 전체 코퍼스에서의 동시 등장 확률이 되도록 만드는 것'**      

이를 만족하도록 임베딩 벡터를 만드는 것이 목표

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20dot%5C%20product%28w_%7Bi%7D%5C%20%5Ctilde%7Bw_%7Bk%7D%7D%29%20%5Capprox%5C%20P%28k%5C%20%7C%5C%20i%29%20%3D%20P_%7Bik%7D)

더 정확히는 

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20dot%5C%20product%28w_%7Bi%7D%5C%20%5Ctilde%7Bw_%7Bk%7D%7D%29%20%5Capprox%5C%20log%5C%20P%28k%5C%20%7C%5C%20i%29%20%3D%20log%5C%20P_%7Bik%7D)

​      

임베딩 벡터들을 만들기 위한 손실 함수를 설계      

GloVe의 연구진들은 벡터 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20w_%7Bi%7D%2C%20w_%7Bj%7D%2C%20%5Ctilde%7Bw_%7Bk%7D%7D)를 가지고 어떤 함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)를 수행하면, ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20P_%7Bik%7D%20/%20P_%7Bjk%7D)가 나온다는 초기 식으로부터 전개 시작        

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F%28w_%7Bi%7D%2C%5C%20w_%7Bj%7D%2C%5C%20%5Ctilde%7Bw_%7Bk%7D%7D%29%20%3D%20%5Cfrac%7BP_%7Bik%7D%7D%7BP_%7Bjk%7D%7D)

이 함수![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)가 어떤 식을 가지고 있는지는 정해진 게 없다      

함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)는 두 단어 사이의 동시 등장 확률의 크기 관계 비(ratio) 정보를 벡터 공간에 인코딩하는 것이 목적      

GloVe 연구진들은 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20w)와 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20%5Ctilde%7Bw%7D)라는 두 벡터의 차이를 함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)의 입력으로 사용하는 것을 제안      

​      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F%28w_%7Bi%7D%20-%5C%20w_%7Bj%7D%2C%5C%20%5Ctilde%7Bw_%7Bk%7D%7D%29%20%3D%20%5Cfrac%7BP_%7Bik%7D%7D%7BP_%7Bjk%7D%7D)

우변은 스칼라값이고 좌변은 벡터값

이를 성립하기 해주기 위해서 함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)의 두 입력에 내적(Dot product)을 수행      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F%28%28w_%7Bi%7D%20-%5C%20w_%7Bj%7D%29%5E%7BT%7D%20%5Ctilde%7Bw_%7Bk%7D%7D%29%20%3D%20%5Cfrac%7BP_%7Bik%7D%7D%7BP_%7Bjk%7D%7D)

선형 공간(Linear space)에서 단어의 의미 관계를 표현하기 위해 뺄셈과 내적      

​        

함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)가 만족해야 할 필수 조건이 있다       

중심 단어 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20w)와 주변 단어 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20%5Ctilde%7Bw%7D)라는 선택 기준은 실제로는 무작위 선택이므로      

이 둘의 관계는 자유롭게 교환될 수 있도록 해야 한다

이것이 성립 하기 위해서 GloVe 연구진은 함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)가 실수의 덧셈과 양수의 곱셈에 대해서 **준동형(Homomorphism)**을 만족하도록 식으로 나타내면 

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F%28a&plus;b%29%20%3D%20F%28a%29F%28b%29%2C%5C%20%5Cforall%20a%2C%5C%20b%5Cin%20%5Cmathbb%7BR%7D)

​      

이 준동형식을 현재 전개하던 GloVe 식에 적용할 수 있도록      

전개하던 GloVe 식에 따르면, 함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)는 결과값으로 스칼라 값![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Ctiny%20%5Cfrac%7BP_%7Bik%7D%7D%7BP_%7Bjk%7D%7D)이  나와야 한다      

준동형식에서 a와 b가 각각 벡터값이라면 함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)의 결과값으로는 스칼라 값이 나올 수 없지만, a와 b가 각각 사실 두 벡터의 내적값이라고 하면 결과값으로 스칼라 값이 나올 수 있다     

그러므로 위의 준동형식을      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20F%28v_%7B1%7D%5E%7BT%7Dv_%7B2%7D%20&plus;%20v_%7B3%7D%5E%7BT%7Dv_%7B4%7D%29%20%3D%20F%28v_%7B1%7D%5E%7BT%7Dv_%7B2%7D%29F%28v_%7B3%7D%5E%7BT%7Dv_%7B4%7D%29%2C%5C%20%5Cforall%20v_%7B1%7D%2C%5C%20v_%7B2%7D%2C%5C%20v_%7B3%7D%2C%5C%20v_%7B4%7D%5Cin%20V)

v1, v2, v3, v4 는 각각 벡터값       

​       

앞서 작성한 GloVe 식에서는 wi와 wj라는 두 벡터의 차이를 함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)의 입력으로 받았다     

GloVe 식에 바로 적용을 위해 준동형 식을 이를 뺄셈에 대한 준동형식으로 변경

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20F%28v_%7B1%7D%5E%7BT%7Dv_%7B2%7D%20-%20v_%7B3%7D%5E%7BT%7Dv_%7B4%7D%29%20%3D%20%5Cfrac%7BF%28v_%7B1%7D%5E%7BT%7Dv_%7B2%7D%29%7D%7BF%28v_%7B3%7D%5E%7BT%7Dv_%7B4%7D%29%7D%2C%5C%20%5Cforall%20v_%7B1%7D%2C%5C%20v_%7B2%7D%2C%5C%20v_%7B3%7D%2C%5C%20v_%7B4%7D%5Cin%20V)

​      

함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)의 우변은 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20F%28%28w_%7Bi%7D%20-%5C%20w_%7Bj%7D%29%5E%7BT%7D%20%5Ctilde%7Bw_%7Bk%7D%7D%29%20%3D%20%5Cfrac%7BF%28w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%7D%7BF%28w_%7Bj%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%7D) 으로 바뀌어야 한다     

우변은 본래 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Ctiny%20%5Cfrac%7BP_%7Bik%7D%7D%7BP_%7Bjk%7D%7D)였으므로, 결과적으로      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfrac%7BP_%7Bik%7D%7D%7BP_%7Bjk%7D%7D%20%3D%20%5Cfrac%7BF%28w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%7D%7BF%28w_%7Bj%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%7D)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20F%28w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%20%3D%20P_%7Bik%7D%20%3D%20%5Cfrac%7BX_%7Bik%7D%7D%7BX_%7Bi%7D%7D)

​      

​        

좌변을 풀어쓰면 

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20F%28w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%5C%20-%5C%20w_%7Bj%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%20%3D%20%5Cfrac%7BF%28w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%7D%7BF%28w_%7Bj%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%7D)

뺄셈에 대한 준동형식의 형태와 정확히 일치      

​        

이제 이를 만족하는 함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)를 찾아야한다      

바로 지수 함수(Exponential function)      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20F)를 지수 함수 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20exp) 로

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20exp%28w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%5C%20-%5C%20w_%7Bj%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%20%3D%20%5Cfrac%7Bexp%28w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%7D%7Bexp%28w_%7Bj%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%7D)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20exp%28w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%29%20%3D%20P_%7Bik%7D%20%3D%20%5Cfrac%7BX_%7Bik%7D%7D%7BX_%7Bi%7D%7D)

두번째 식으로부터       

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%20%3D%20log%5C%20P_%7Bik%7D%20%3D%20log%5C%20%28%5Cfrac%7BX_%7Bik%7D%7D%7BX_%7Bi%7D%7D%29%20%3D%20log%5C%20X_%7Bik%7D%20-%20log%5C%20X_%7Bi%7D)

여기서 상기해야할 것은  ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20w_%7Bi%7D)와 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20%5Ctilde%7Bw_%7Bk%7D%7D) 는 두 값의 위치를 서로 바꾸어도 식이 성립해야 한다는 것이다      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20X_%7Bik%7D) 는 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20X_%7Bki%7D) 와 같다     

걸림돌은 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20log%5C%20X_%7Bi%7D)       

그래서 GloVe 연구팀은 이 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20log%5C%20X_%7Bi%7D)항을 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20w_%7Bi%7D)에 대한 편향 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20b_%7Bi%7D)라는 상수항으로 대체     

같은 이유로 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20%5Ctilde%7Bw_%7Bk%7D%7D)에 대한 편향 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20%5Ctilde%7Bb_%7Bk%7D%7D)를 추가      

​       

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20w_%7Bi%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bk%7D%7D%20&plus;%20b_%7Bi%7D%20&plus;%20%5Ctilde%7Bb_%7Bk%7D%7D%20%3D%20log%5C%20X_%7Bik%7D)

__이 식이 손실 함수의 핵심이 되는 식__      

__우변의 값과의 차이를 최소화는 방향으로 좌변의 4개의 항은 학습을 통해 값이 바뀌는 변수들이 된다__       

손실 함수는 다음과 같이 일반화될 수 있다      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20Loss%5C%20function%20%3D%20%5Csum_%7Bm%2C%20n%3D1%7D%5E%7BV%7D%5C%20%28w_%7Bm%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bn%7D%7D%20&plus;%20b_%7Bm%7D%20&plus;%20%5Ctilde%7Bb_%7Bn%7D%7D%20-%20logX_%7Bmn%7D%29%5E%7B2%7D)

V : 단어 집합의 크기      

아직 최적의 손실 함수라기에는 부족       

​      

 GloVe 연구진은 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20log%5C%20X_%7Bik%7D)에서 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20X_%7Bik%7D)값이 0이 될 수 있음을 지적      

대안 중 하나는 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20log%5C%20X_%7Bik%7D)항을 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20log%5C%281%20&plus;%20X_%7Bik%7D%29) 로 변경하는 것      

하지만 이렇게 해도 여전히 해결되지 않는 문제가 있다      

바로 동시 등장 행렬 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20X)는 마치 DTM처럼 희소 행렬(Sparse Matrix)일 가능성이 다분하다는 점       

많은 값이 0이거나, 동시 등장 빈도가 적어서 많은 값이 작은 수치를 가지는 경우가 많다      

GloVe의 연구진은 동시 등장 빈도의 값![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20X_%7Bik%7D)이 굉장히 낮은 경우 정보에 거의 도움이 되지 않는다고 판단       

​       

GloVe 연구팀이 선택한 것은 바로 ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20X_%7Bik%7D)의 값에 영향을 받는 가중치 함수(Weighting function) ![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20f%28X_%7Bik%7D%29)를 손실 함수에 도입하는 것       

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20f%28X_%7Bik%7D%29)의 그래프      

![image](https://wikidocs.net/images/page/22885/%EA%B0%80%EC%A4%91%EC%B9%98.PNG)

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20X_%7Bik%7D)의 값이 작으면 상대적으로 함수의 값은 작도록 하고, 값이 크면 함수의 값은 상대적으로 크도록      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20X_%7Bik%7D)가 지나치게 높다고해서 지나친 가중치를 주지 않기 위해서 또한 함수의 최대값이 정해져있다     

이 함수의 값을 손실 함수에 곱해주면 __가중치의 역할__ 을 할 수 있다       

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20f%28x%29%20%3D%20min%281%2C%5C%20%28x/x_%7Bmax%7D%29%5E%7B3/4%7D%29)

​      

최종적으로 일반화 된 손실 함수      

![image](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csmall%20Loss%5C%20function%20%3D%20%5Csum_%7Bm%2C%20n%3D1%7D%5E%7BV%7D%5C%20f%28X_%7Bmn%7D%29%28w_%7Bm%7D%5E%7BT%7D%5Ctilde%7Bw_%7Bn%7D%7D%20&plus;%20b_%7Bm%7D%20&plus;%20%5Ctilde%7Bb_%7Bn%7D%7D%20-%20logX_%7Bmn%7D%29%5E%7B2%7D)