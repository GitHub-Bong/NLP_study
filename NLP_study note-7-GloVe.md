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

<img src = "/image/GloVe 1.PNG" width = "600px">