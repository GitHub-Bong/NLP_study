# Attention Is All You Need 논문 리뷰       

2017년도 구글 팀이 발표한 논문.      

이전까지는 RNN 과 Attention Mechanism  같이 활용하는 기법이 많이 사용되었음.

논문에서는 RNN CNN 사용하지 않고 Attention만 사용해서 기계번역 task에서 좋은 성능을 거두었다.        

<br/>

- [1 Abstract](#Abstract)
- [2 Introduction](#Introduction)
- [3 Model Architecture](#Model-Architecture)
  - [3.1 Encoder and Decoder Stacks](#Encoder-and-Decoder-Stacks)
  - [3.2 Attention](#Attention)
    - [3.2.1 Scaled Dot-Product Attention](#Scaled-Dot-Product-Attention)
    - [3.2.2 Multi-Head Attention](#Multi-Head-Attention)
    - [3.2.3 Applications of Attention in our Model](#Applications-of-Attention-in-our-Model)
  - [3.3 Position-wise Feed-Foward Networks](#Position-wise-Feed-Foward-Networks)
  - [3.4 Embeddings and Softmax](#Embeddings-and-Softmax)
  - [3.5 Positional Encoding](#Positional-Encoding)
- [4 Why Self-Attention](#Why-Self-Attention)
- [5 Training](#Training)
  - [5.1 Training Data and Batching](#Training-Data-and-Batching)
  - [5.2 Hardware and Schedule](#Hardware-and-Schedule)
  - [5.3 Optimizer](#Optimizer) 
  - [5.4 Reglarization](#Reglarization)
- [6 Results](#Results)
  - [6.1 Machine Translation](#Machine-Translation)
  - [6.2 Model Variations](#Model-Variations)
  - [6.3 English Constituency Parsing](#English-Constituency-Parsing) 
- [7 Conclusion](#Conclusion) 



# Abstract          

이전까지는 Encoder와 Decoder를 포함하는 형태로 복잡한 RNN 혹은 CNN 기반의 시쿼스간 변형(ex. 기계번역)이 이루어지는 모델들이 지배적이었다.        

가장 좋은 성능을 보인 모델 역시 Encoder와 Decoder에 Attention을 활용했다.         

여기서 필자는 __Transformer__ 라는 아키텍쳐 제안했다.    

이는 __RNN CNN 없이 전적으로 Attention Mechanism만 기반으로__ 했다.

__행렬곱을 사용하기 때문에 병렬적으로 처리해__  훨씬 빠른 속도로 훈련이 가능했다.       

WMT 2014 영어를 독일어로, 영어를 불어로 번역하는 task에서 압도적인 성능을 보였다. SOTA      

기계 번역 뿐만 아니라 다른 task에도 일반화가 잘 되는 것을 보였다.         

​      

<br/>

# Introduction        

RNN, LSTM, GRU 등의 모델들이 sequence modeling과 sequence transduction 문제들에서 사용되어왔다.     

RNN 모델들은 recurrent하게 __토큰의 갯수만큼 입력을 넣어야 하기 때문에__  메모리 부족 문제를 야기시킨다.     

Attention mechanism은 RNN과 결합해 사용되어왔다.     

본 논문에서는 __Attention mechaism만 사용하고 순차적으로 입력하는 것이 아니라 병렬적인 처리를 통해__  훨씬 좋은 성능을 보였다.     

​      

<br/>

# Background     

__Self-Attention__ 가끔 intra-attention이라고도 불리는 메커니즘은       

__하나의 시퀀스에서 다른 위치에 있는 단어들끼리 서로 가중치를 계산해 시퀀스를 잘 표현하도록 하는 메커니즘__이다.      

Reading comprehension, abstractive summarization, textual entailment 등과 같은 task들에서 성공적으로 사용되고 있다.     

Transformer는 RNN이나 CNN을 사용하지 않고 전적으로 self-attention으로 sequence representation을 연산하는 최초의 모델이다.       

​          

<br/>

# Model Architecture        

시퀀스 변형 모델들은 인코더-디코더 구조를 갖추고 있다.      

(x1, ... , xn) 의 input sequence를 (z1, ... ,zn) 형태로 나타내고 (y1, ... , ym) 의 m개의 토큰을 가진 시퀀스를 출력했다.     

각 단계마다 모델은 auto-regressive하게 이전 단계들의 symbols을 통해 다음에 출력 될 것을 만들었다.     

Transformer도 __마찬가지로 인코더-디코더 구조__ 를 갖추고 있다.    

__Self-attention__  과 __point-wise__ 를 쌓은 구조를 인코더와 디코더 모두에서 갖추고 있다.     

 ![image](https://kh-kim.gitbooks.io/pytorch-natural-language-understanding/content/assets/nmt-transformer-1.png)



<br/>

## Encoder and Decoder Stacks       

__Encoder__ :      

논문에서는 __6개의 동일한 Layer __ 로 구성      

각 Layer은 __2 sub-layers__ 로 구성 

첫 번째 sub-layer은 __multi-head self-attention mechanism__     

두 번째 second-layer은 __position-wise fully connected feed-foward network__      

각 sub-layer마다 __residual connection__ 과 __layer normalization__ 을 거쳐 __LayerNorm(x + Sublayer(x))__ 이라는 출력을 얻게 된다.      

모든 층들의 결과는 임베딩 층의 차원과 __동일한 차원__을 갖게 된다. (논문에서는 512차원)      

​        

__Decoder__ :        

논문에서는 __6개의 동일한 Layer __ 로 구성      

디코더에는 인코더의 출력을 가지고 attention mechanism을 수행하는 __multi-head attention__ 이라는 sub-layer가 있다.       

인코더와 비슷하게 __residual connection__ 과 __layer normalization__ 을 수행한다.     

__현재 예측해야하는 시점 이후의 단어들을 참고하지 못하게 하도록 masking을 하는 점__ 이 인코더와 다르다.      

​        

<br/>

## Attention        

__각각의 query가 key에 대해 어텐션을 수행하는 메커니즘__           

   

![image](https://kh-kim.gitbooks.io/pytorch-natural-language-understanding/content/assets/nmt-transformer-2.png)

​        

### Scaled Dot-Product Attention      

query와 key들의 차원은 ![image](https://latex.codecogs.com/gif.latex?d_%7Bk%7D)       

value들의 차원은 ![image](https://latex.codecogs.com/gif.latex?d_%7Bv%7D)            

query와 모든 key들에 대해 __내적__ 을 수행한다.       

각각을 ![image](https://latex.codecogs.com/gif.latex?%5Csqrt%7Bd_%7Bk%7D%7D)  에 대해 __나눠준다.__     

__softmax 함수__ 를 적용해 확률값을 구하고 __value 와 곱해준다.__        

​        

__실제로는 한번에 query와 key들을 행렬 형태로 묶어서 병렬적으로 연산한다.__      

![image](https://latex.codecogs.com/gif.latex?Attention%28Q%2C%20K%2C%20V%29%20%3D%20softmax%28%7BQK%5ET%5Cover%7B%5Csqrt%7Bd_k%7D%7D%7D%29V)

​        

내적을 수행하는 dot-product attention은        

query와 key가 특정한 행렬곱에 함께 입력되는 형태로 동작하는 additive attention보다 훨씬 빠르고 공간 효율적이다.      

그러나 ![image](https://latex.codecogs.com/gif.latex?%5Csqrt%7Bd_%7Bk%7D%7D) 을 사용하지 않는 경우 성능이 더 안 좋았다.      

논문은 softmax 함수는 값이 너무 큰 경우 gradients가 매우 작아지는데 학습이 잘 되기 위해 ![image](https://latex.codecogs.com/gif.latex?%5Csqrt%7Bd_%7Bk%7D%7D) 

로 나누어 값을 작게 만들었다.      

​       

​     

### Multi-Head Attention       

8개의 어텐션 수행한  head들은 다시 연결하게 되는데  이로써 __입력과 출력의 차원이 동일__ 해졌다.     

![image](https://media.vlpt.us/images/changdaeoh/post/fbbb9832-272d-40e2-9a10-628c5e9c8def/image.png)

​       

논문에서는 value의 차원을 query나 key와 똑같이 맞출 필요는 없지만 동일하게  (512/h) 차원으로 만들었다.     h:head의 갯수

![image](https://latex.codecogs.com/gif.latex?h%3D8%2C%20%5C%3A%20%5C%3A%20d_k%20%3D%20d_v%20%3D%20d_%7Bmodel%7D/h%20%3D%2064)

​             

​           

### Applications of Attention in our Model        

Transformer의 3가지 multi-head attention은 조금씩 다르다      

- 'encoder-decoder attention' 에서는 __이전 decoder layer의 출력을 query로, encoder 의 최종 출력을 key, value로__  가지고 온다.      
- 인코더에 self-attention layers가 있는데 __query, key, value 를 모두 동일하게 이전 layer 출력에서__  가지고 온다.       
- 디코더에도 self-attention layers가 있는데 __현재 예측해야할 단어 이후의 정보를 참고하지 못하도록 scaled dot-product attention에서 maksing 을 실시__ 해 softmax함수를 거쳤을 때 0에 매우 가까운 값을 갖도록 한다.         

​       

<br/>

## Position-wise Feed-Foward Networks        

Attention sub-layers에 추가적으로, 인코더와 디코더는 fully connected feed-forward network를 갖추고 있다.    

__2 linear transformations 사이 RELU activation이__  사용되었다.     

![image](https://latex.codecogs.com/gif.latex?FFNN%28x%29%20%3D%20MAX%280%2C%20x%7BW_%7B1%7D%7D%20&plus;%20b_%7B1%7D%29%7BW_2%7D%20&plus;%20b_2)

2 linear transformations의 가중치 행렬들은 다른 파라미터 값을 사용한다.     

kernel size가 1인 두 convolution 층으로 이해할 수도 있다.     

​      

attention sub-layers와 마찬가지로 __인풋과 아웃풋의 차원은 동일__ 하고,     

inner-layer의 차원은 논문에서는 ![image](https://latex.codecogs.com/gif.latex?d_%7Bff%7D) = 2048            

​     

<br/>

## Embeddings and Softmax       

기존의 다른 sequence transduction models 처럼,  임베딩 층을 통해 임베딩 차원 ![image](https://latex.codecogs.com/gif.latex?d_%7Bmodel%7D)   이 된다.

__linear transformation 과 softmax 함수를 통해__ 디코더의 아웃풋으로부터 다음 단어를 예측한다.  

임베딩 layers에서는 가중치에 ![image](https://latex.codecogs.com/gif.latex?%5Csqrt%7Bd_%7Bmodel%7D%7D) 를 나눠준다.      

​         

<br/>

## Positional Encoding         

논문의 모델은 recurrence와 convolution을 사용하지 않았기 때문에 __단어의 위치 정보를 임베딩에 같이 넣어줄 필요__ 가 있었다.        

![image](https://latex.codecogs.com/gif.latex?PE_%7B%28pos%2C%5C%202i%29%7D%3Dsin%28pos/10000%5E%7B2i/d_%7Bmodel%7D%7D%29)

![image](https://latex.codecogs.com/gif.latex?PE_%7B%28pos%2C%5C%202i&plus;1%29%7D%3Dcos%28pos/10000%5E%7B2i/d_%7Bmodel%7D%7D%29)

pos  :  입력 문장에서 임베딩 벡터의 위치        

i  :  임베딩 벡터 내의 차원의 인덱스     

​      

이러한 형태가 아닌 학습이 가능한 positional embeddings를 사용해봤는데 성능은 거의 동일했다.

필자는 학습과정에서 사용된 sequence보다 더 긴 sequence가 들어왔을 때 이런 정현파 함수가 더 좋기 때문에 선택했다.

​    

<br/>

<br/>

# Why Self-Attention         

Self-Attention layer를 recurrent와 convoluntional layer와 비교하면서 3가지 바라는 목표가 있었다.     

- 첫 번째는 __줄어든 계산 복잡도__     
- 두 번째는  __병렬적 연산을 통해 줄어든 계산의 양__
- 마지막은 __long-range dependency를 잘 처리__     

 ![image](https://greeksharifa.github.io/public/img/2019-08-17-Attention%20Is%20All%20You%20Need/03.png) 

n은 sequence 길이인데 임베딩 차원인 d보다는 작은 경우가 대부분이다. 그러므로 Self-Attention은 복잡성에서도 훨씬 효율적이다.     

한 번의 병렬적인 연산을 하기 때문에 O(1)      

네트워크에서 횡단할 수 있는 거리가 long-term dependency를 잘 학습하게끔 하는 요인인데 최대 길이를 비교해봤다.       

Self-Attention(restricted)은 굉장히 긴 sequences를 연산하는 과정을 향상시키기 위해서 해당 단어 근처 r 만큼만 활용하게 되는 방법이다.        

Convolution은 kernel size k가 n이어도 복잡성이 self-attention layer와 point-wise feed-forward layer를 결합한 것과 같다.     

​       

self-attention은 모델을 좀 더 설명하기 쉽게 해줬다.    

softmax 함수를 적용해 얻은 __attention distribution을 보게 되면__ 해당 단어가 어떤 단어의 영향을 많이 받았는지 알 수 있다.     

![image](http://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png) 



​               

<br/>

<br/>

# Training       

​       

### Training Data and Batching      

WMT 2014 English-German dataset (byte-pair encoding으로 인코딩된 450만 문장들)과     

WMT 2014 English-French dataset (word-piece 알고리즘으로 인코딩된 3600만 문장들)을 사용했다.                 

각 training batch는 25000개의 토큰을 담고 있다.       

​        

​       

### Hardware and Schedule      

8 NVIDIA P100 GPUs를 사용해 모델을 학습했다.     

baseline model을 학습하는데 12시간이 걸렸다.     

​        

​           

### Optimizer           

__Adam Optimizer를__ 사용하였다.

learning rate를 warmup_steps(논문에서는 4000)만큼의 training steps 동안 증가시키고 그 후에는 감소 시켰다.      

​          

​          

### Reglarization        

__Residual Dropout__        

각 Sub layer의 출력에 __dropout을 적용__ 했다.       

인코더와 디코더에서 embeddings과 positional encodings의 __합에 dropout을 적용__ 했다.     

dropout 비율 = 0.1      



__Label Smoothing__       

학습 과정에서 __출력 값에 label smoothing을 적용해__  예측값을 너무 과신하지 않게 했다.     

![image](https://latex.codecogs.com/gif.latex?%5Cepsilon%20_%7Bls%7D%20%3D%200.1)

이는 accuracy와 BLEU score를 향상시켰다.         

​        

<br/>

<br/>

# Results        

__Machine Translation__       

![image](https://d3i71xaburhd42.cloudfront.net/204e3073870fae3d05bcbc2f6a8e263d9b72e776/8-Table2-1.png)

big transformer model은 기존의 SOTA 모델보다 훨씬 좋은 성능을 보였다.     

base model도 기존 모델들보다 더욱 훈련 시간은 짧고 좋은 성능을 보였다.     

​      

​         

__Model Variations__         

Transformer의 다양한 하이퍼 파라미터들의 중요도를 측정하기 위해 English-to-German 데이터를 사용해 다양하게 바꿔가며 측정했다.     

![image](https://d3i71xaburhd42.cloudfront.net/204e3073870fae3d05bcbc2f6a8e263d9b72e776/9-Table3-1.png)

(A)  : head의 수를 바꿔봤다. 그에 따라 ![image](https://latex.codecogs.com/gif.latex?d_k%2C%20%5C%3A%20d_v)  가 바뀌었다. 8일때 가장 성능이 좋았다.      

(B) : ![image](https://latex.codecogs.com/gif.latex?d_k) 의 수만 증가했는데 성능은 더 안좋아졌다.        

(C), (D) : 모델이 더 커짐에 따라 성능이 더 좋아졌고 dropout은 over-fitting을 방지하는데 도움이 됐다.      

(E) : 원래 사용했던 sinusodial positional encoding 대신 학습 가능한 다른 positional encoding을 사용해봤지만 성능은 거의 동일했다.       

​           

​       

### English Constituency Parsing      

기계 번역 뿐만 아니라 __영어 구문 분석 분야에서__  실험했을 때 task-specific tuning 이 부족했지만서도   

RNN Grammar을 제외한 모든 이전 모델들보다 좋은 성능을 보였다.         

인풋보다 아웃풋이 길기 때문에 maximum output length를 input length 보다 300 길게 했다.       

![image](https://d3i71xaburhd42.cloudfront.net/204e3073870fae3d05bcbc2f6a8e263d9b72e776/9-Table4-1.png)

​       

<br/>

<br/>

# Conclusion        

__Transformer 모델은 recurrent layers을 전혀 사용하지 않고 attention을 기반으로한 multi-headed self-attention으로 encoder-decoder 구조를 갖춘 첫 번째 모델이다.__       

번역 task에서 확실히 빠르게 훈련이 가능했고 새로운 SOTA가 되었다.    

Transformer를 task 뿐만 아니라 다른 task에서도 사용 가능하도록 계획했다.

​      

[The code is available at here](#https://github.com/tensorflow/tensor2tensor#sentiment-analysis)









