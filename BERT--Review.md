# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding 논문 리뷰      

2018년 구글이 공개한 논문.      

일부 성능 평가에서 인간보다 더 높은 정확도를 보였다.     

BERT는 NLP task를 교육 없이 양방향으로 사전학습하는 첫 시스템     



<br/>

- [Abstract](#Abstract)
- [1 Introduction](#1-Introduction)
- [2 Related Work](#2-Related-Work)
  - [2.1 Unsupervised Feature-based Approaches](#2-1-Unsupervised-Feature-based-Approaches)
  - [2.2 Unsupervised Fine-tuning Approaches](#2-2-Unsupervised-Fine-tuning-Approaches)
  - [2.3 Transfer Learning from Supervised Data](#2-3-Transfer-Learning-from-Supervised-Data)
- [3 BERT](#3-BERT)
  - [3.1 Pre-training BERT](#3-1-Pre-training-BERT)
- [4 Experiments](#4-Experiments)
  - [4.1 GLUE](#4-1-GLUE)
  - [4.2 SQuAD v1.1](#4-2-SQuAD-v11)
  - [4.3 SQuAD v2.0](#4-3-SQuAD-v20)
  - [4.4 SWAG](#4-4-SWAG)
- [5 Ablation Studies](#5-Ablation-Studies) 
  - [5.1 Effect of Pre-training Tasks](#5-1-Effect-of-Pre-training-Tasks)
  - [5.2 Effect of Model Size](#5-2-Effect-of-Model-Size)
  - [5.3 Feature-based Approach with BERT](#5-3-Feature-based-Approach-with-BERT)
- [6 Conclusion](#6-Conclusion)

​         

<br/>

# Abstract      

__BERT__ 란?   

__B__ idirectional     

__E__ ncoder      

__R__   epresentations from      

__T__ ransformers         

​     

BERT는 모든 층에서 양쪽의 문맥을 이용해 특징을 표현하는 deeply bidirectional representations을 pretrain하게끔 디자인 되었다.    

그 결과 하나의 추가적인 출력층으로 다양한 task에서 SOTA 모델들을 만들 수 있었다.     

BERT는 11개의 NLP tasks에서 SOTA 모델을 달성했다.      

​       

# 1 Introduction        

Transfer Learning 에는       

__feature-based approach__ 와 __fine-tuning approach__ 가 있다.     

Feature-based approach는 pre-trained 된 representations들을 추가적으로 사용하는 방식이다. (Ex. ELMo)      

Fine-tuning approach는 task-specific한 parameters는 최소로 사용하고 pre-trained된 parameters를 모두 fine-tuning하는 방식이다.  (EX. OpenAI GPT)

두 방식 모두 pre-training 동안 단방향(unidirectional) 언어 모델을 사용한다.     

이러한 점은 특히 fine-tuning based approach을 양 방향의 문맥을 통합해야 하는 token-level task에 적용할 때 심각한 문제가 된다.           

BERT는 이전의 문제점들을 'masked language model' __(MLM)__ 을 사용하여 완화했다.       

MLM은 인풋의 일부를 랜덤으로 마스킹하고 문맥을 사용해 원래의 토큰을 예측하는 것이 목적이다.       

MLM은 양 방향의 문맥을 함께 결합시켜 사용하는 것이 기존의 단 방향 언어 모델들과 다르다.      

추가적으로, next setence prediction __(NSP)__ 을 이용했다.      

NSP는 두 문장이 이어지는 문장이 아닌지를 예측하는 것이 목적이다.        

<br/>

# 2 Related Work        

​       

## 2-1 Unsupervised Feature-based Approaches       

ELMO는 left-to-right 과 right-to-left를 통해 얻은 token representations을 합쳐 기존의 임베딩 벡터와 연결해 사용했는데 여러 주요 NLP task에서 SOTA를 기록했다.       

​        

## 2-2 Unsupervised Fine-tuning Approaches      

feature-based approaches처럼, 처음에는 워드 임베딩 벡터만 pre-train했었다. 그 이후 최근에는, pre-trained된 파라미터들을 fine-tuning했는데 이러한 방식은 파라미터 수가 적은 장점이 있었다.     

Open AI GPT는 이러한 장점을 살려 많은 sentence-level tasks에서 SOTA를 기록했다.        

​          

## 2-3 Transfer Learning from Supervised Data        

대용량의 사전학습된 모델을 transfer learning을 하는 것이 중요하다는 것을 Computer Vision 연구에서 보여줬다.       

특히 기계 번역이나 추론 task에서 좋은 성능을 보였다.      

​       

<br/>

# 3 BERT         

BERT에는 크게 두 가지 스텝이 있다.     

__pre-training__ 과 __fine-tuning__      

pre-training 동안 __unlabeled__ data를 사용해 훈련되었고      

pre-trained된 모든 parameters를 labeled data를 통해 fine-tuning했다.       

​      

BERT의 주요 특징은 task가 다르더라도 통일된 구조를 갖는다는 것이었다.       

pre-training할 때의 구조와 task에 맞춘 구조에는 최소의 차이가 존재했다.         

​           

![image](https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F6fe12b1e-8a0c-46bc-a221-db2bf167908e%2FUntitled.png&blockId=81567ecd-8101-4119-8aaa-42cf1ff53a98)



__Model Architecture__       

BERT 모델의 구조는 __multi-layer bidirectional Transformer encoder__         

L : number of layers      

H : hidden size     

A : number of self-attention heads      

__BERT BASE__  :  (L=12, H=768, A=12, Total Parameters=110M)          

__BERT LARGE__ : (L=24, H=1024, A=16, Total Parameters=340M)        

BERT BASE 모델은 Open AI GPT와 비교하기 위해 같은 모델 크기를 선택했다.      

__BERT Transformer는 양방향 self-attention을 이용했으나 Open AI GPT는 left-to-right 단방향 self-attention을 이용했다.__              

​         

__Input/Output Representations__         

BERT가 다양한 task를 다룰 수 있게 input representation을 한 문장이나 문장 여러개를 모두 하나의 token sentence로 사용했다.         

BERT의 input token sequence를 __'sequence'__ 라 지칭했다.       

이는 __하나의 문장일 수도 또는 두 문장이 결합되어있는 상태일 수 도 있다.__         

30,000 token을 활용해 WordPiece 임베딩을 사용했다.     

모든 sequence의 처음에는 __[CLS] 토큰을 위치시켰고__ 이에 해당하는 최종 은닉 상태는 분류 task에 활용했다.    

두 가지 방법으로 문장들을 구분했다.    

첫 번째로는 문장 사이에 __[SEP] 토큰을 위치시켰다.__       

두 번째로는 learned embedding을 추가해 token이 A문장에 속하는지 B문장에 속하는지를 나타냈다.             

![image](https://mblogthumb-phinf.pstatic.net/MjAyMDA3MDNfMjk5/MDAxNTkzNzEyMTc4OTAz.7_x4-VCrlkQmMraxFLmBGXjrO2SgX_-aO9idCBsK_JMg.cbW7Gnnaiu-G3m-rkhRGTOs_N4Dct1RVpr7Tx_SW71Qg.PNG.winddori2002/4.PNG?type=w800)



token에 대해 해당 token의 __WordPiece Embedding, Segment Embedding, Position Embedding을__ 합친 것이 input representation이 되도록 했다.         

​           

## 3-1 Pre-training BERT        

two unsupervised tasks로 BERT를 pre-train시켰다.           

​          

__Task #1: Masked LM__            

BERT 모델의 bidirectional conditioning은 간접적으로 단어들을 볼 수 있기 때문에 모델이 당연하게 target word를 예측할 수 있게 됐다.       

그렇기 때문에 input token에 일정 비율로 masking을 진행하는 __masked LM(MLM)__ 을 진행했다.       

랜덤으로 15%의 token을 마스킹했는데 이는 pre-training과 fine-tuning 때 mismatch를 만들었다. fine-tuning 때는 [MASK] 토큰이 등장하지 않기 때문이다.       

__A.1 Illustration of the Pre-training Tasks 참고__      

이러한 점을 완화하기 위해 랜덤으로 선택된 token 중 __80%의 token들은 [MASK]로, 10%의 token들은 랜덤 token으로, 10%의 token들은 동일하게 두었다.__          

input token의 마지막 은닉 상태 벡터는 cross entropy loss를 이용해 원래의 token을 예측하는데 사용되었다.         

​       

__Task #2: Next Sentence Prediction (NSP)__           

sentence relationships을 훈련시키기 위해  __50%는 실제 다음 문장을 가져오고 50%는 랜덤 문장__ 을 가져와 pre-train했다.        

__A.1 Illustration of the Pre-training Tasks 참고__       

그림처럼 [CLS] 토큰의 최종 은닉 벡터가 NSP에 사용되었다.       

간단함에도 불구하고 Question Answering이나 Natural Language Inference에서 매우 좋은 성능을 보였다.         

​                   

__Pre-training data__           

BooksCorpus (800M words)와 English Wikipedia (2,500M words)가 사전 학습에 사용되었다.     

​        

​      

## 3-2 Fine-tuning BERT          

![image](https://mblogthumb-phinf.pstatic.net/MjAyMDA3MDNfNTIg/MDAxNTkzNzg1MjE0MzY3.rduTNdjLLygPJdrYnilnGU-UuqOY432HXxyNCfPmVpcg.umGNmf1KEpWC5BvP97H4DJI9hkfOmGd6jnhYKrhLVlIg.PNG.winddori2002/5.PNG?type=w800)

​        

token level tasks (sequence tagging, question answering) 에서는 각 token의 최종 은닉 벡터가 활용되었고        

sequence level tasks (entailment, sentiment analysis) 에서는 [CLS] 토큰의 출력 값이 활용되었다.            



pre-training과 비교했을 때, fine-tuning은 학습 시간이 매우 적게 걸렸다. 

​    

<br/>

# 4 Experiments          

BERT를 11 NLP tasks에 fine-tuning한 결과         

​           

## 4-1 GLUE          

General Language Understanding Evaluation(GLUE)는 diverse natural language understanding collection이다.        

[CLS] 토큰의 최종 은닉 벡터를 사용했다.    

분류해야할 label 수를 k라 했을 때, K x H(hidden size) 크기의 classification layer만 추가 되었다.     

loss를 ![image](https://latex.codecogs.com/gif.latex?log%28softmax%28CW%5ET%29%29) 로 계산했다.        

![image](https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F166dfe6c-97a1-4405-aeec-e41d8ba051f3%2FUntitled.png&blockId=b6c43d5a-285e-4bcb-a102-5eb79139d3b6)

BERT BASE와 BERT LARGE 모델 모두 모든 task에서 이전 SOTA 보다 좋은 성능을 보였다. BERT LARGE는 BERT BASE 보다 특히 작은 training data에서 좋은 성능을 보였다.         

​          

## 4-2 SQuAD v1.1       

Standford Question Answering Dataset(SQuAD v1.1)은 100k의 question/answer pairs를 모아놓은 데이터 셋이다.      

질문과 Wikipedia의 정답을 포함한 글로 정답을 예측하는 task이다.      

질문은 A embedding, 문단은 B embedding으로 하나의 single packed sequence를 만들었다.      

정답의 첫번째 벡터를 S, 마지막 벡터를 E 이 두 벡터로 fine-tuning 했다.        

어떠한 단어 i가 정답의 첫 번째 단어일 확률은 S와 내적을 한 결과다.  이 결과를 softmax 함수를 통해 가장 큰 값을 찾아냈다.        

마지막 벡터와도 유사하게 계산했다.       

S부터 E까지가 정답일 score는 ![image](https://latex.codecogs.com/gif.latex?S%5Ccdot%20T_%7Bi%7D%20&plus;%20E%5Ccdot%20T_%7Bj%7D) 로 계산해 가장 큰 조합을 predict하는 방식으로 fine-tuning 했다.          

​             

## 4-3 SQuAD v2.0        

SQuAD v2.0은 좀 더 현실성 있도록 단락에 정답이 없는 경우를 만든 데이터 셋이다.        

정답의 예상 구간을 [CLS] 토큰 위치 까지 확장시켰다.       

정답이 없는 경우의 score를 ![image](https://latex.codecogs.com/gif.latex?s_%7Bnull%7D%20%3D%20S%5Ccdot%20C%20&plus;%20E%5Ccdot%20C)

이를 ![image](https://latex.codecogs.com/gif.latex?S%5Ccdot%20T_%7Bi%7D%20&plus;%20E%5Ccdot%20T_%7Bj%7D) 와 비교해      

![image](https://latex.codecogs.com/gif.latex?s_%7Bnull%7D&plus;%5Ctau) 보다 클 경우에 정답이 있다고 예측했다.         

​            

## 4-4 SWAG         

Situations with Adversarial Generations(SWAG) 데이터 셋은 113k개의 sentence-pair  로 구성되어있다.       

문장이 주어졌을 때 4가지 문장 중에 가장 그럴듯하게 이어지는 문장을 고르는 task이다.        

SWAG 데이터 셋에 fine-tuning 하기 위해, 주어진 문장에 가능한 4 문장들을 각각 결합했다.       

task-specific한 parameter를 만들어 [CLS] 토큰의 최종 은닉 벡터와 내적해 나온 score를 softmax 함수를 이용해 normalize 했다.    



<br/>

# 5 Ablation Studies          

​     

## 5-1 Effect of Pre-training Tasks       

masked LM (MLM)은 사용하지만 next sentence prediction을 사용하지 않는 __NO NSP__ 와      

bidirectional을 사용하지 않고 left-context-only만 사용하고 마찬가지로 next sentence prediction을 사용하지 않는 __LTR(Left-to-Right) & No NSP__ 을 이용해 비교해봤다.      

![image](https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F0276744b-d2f3-4c3a-acbd-3939253fdb9b%2FUntitled.png&blockId=bcdb7d1a-c03c-4999-b8f2-e058227dfeba) 

NSP를 사용하지 않았을 때 기존의 BERT BASE 보다 성능이 확연히 떨어진 것을 볼 수 있었다.      

bidirectional 을 사용하지 않았을 때 역시 사용했을 때 보다 성능이 떨어졌다.

SQUAD 데이터 셋에서는 LTR 모델이 right-side context가 은닉 상태에 없기 때문에 성능이 나쁠 것을 쉽게 알 수 있었다. 그래서 BiLSTM을 붙였을 때 그나마 성능이 좋았지만 여전히 다른 pre-trained bidirectional models 보다 성능이 매우 안 좋은 것을 알 수 있었다. 

​            

## 5-2 Effect of Model Size          

layers, hidden units, attention heads 의 수를 바꿔가면서 성능을 비교해봤다.      

![image](https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fdcf884e4-3e1b-4735-8933-0e47d035aaf3%2FUntitled.png&blockId=de5ac463-41ad-4329-b4f8-1a692494b3f6)



모델의 크기가 커질수록 성능이 좋아지는 것을 확인했다.      

특히 모델이 충분히 pre-trained되었기 때문에 매우 작은 scale의 tasks에서도 크기가 클수록 모델의 성능이 향상됨을 확인했다.       

​              

## 5-3 Feature-based Approach with BERT         

모든 task가 Transformer encoder 구조로 쉽게 표현되지 않고,       

computational benefits을 위해    

Featured-based approach을 계속 사용했던 fine-tuning approach와 비교해봤다.      

fine-tuning하지 않은 한 개 이상의 layer를 input으로 하는 BiLSTM을 분류 층 전에 사용해 feature-tuning approach를 실험했다.         

BERT는 feature-based approach와 fine-tuning approach 모두 좋은 성능을 보이는 것을 확인했다.       

![image](https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Ff5b293b9-166b-4b13-817e-cda468524c9c%2FUntitled.png&blockId=e0459314-e4c6-402b-ab63-80b4303f892d)

​         

​         

​            

# 6 Conclusion        

transfer learning 으로 성능이 향상된 최근 언어 모델들을 보면 unsupervisied pre-training 이 핵심적인 부분인 것을 알 수 있었다.     

deep unidirectional 에서 deep bidirectional 구조로 일반화하여 더욱 다양한 NLP task에 활용할 수 있게 하는 것이 저자의 공헌이다.

​          

<br/>



## A.4 Comparison of BERT, ELMo and OpenAI GPT        

![image](https://wikidocs.net/images/page/35594/bert-openai-gpt-elmo-%EC%B6%9C%EC%B2%98-bert%EB%85%BC%EB%AC%B8.png)

BERT와 OpenAI GPT는 finetuning approaches    

ELMo는 feature-based approach       

GPT와 BERT의 가장 큰 차이점은 GPT는 left-to-right Transformer LM을 훈련시킨다는 것이다.      

Bi-directionality와 MLM, NSP 말고도 GPT와 BERT의 차이점  

- GPT는 BooksCorpus (800M)로 , BERT는 BooksCorpus (800M)와 Wikipedia(2,500M) 로 훈련했다.      
- GPT는 [SEP]토큰과 [CLS]토큰을 fine-tuning 때만 사용했지만 BERT는 [SEP]와 [CLS] 토큰 그리고 A문장, B문장으로 나누는 embedding을 pre-training 부터 사용했다.      







  



