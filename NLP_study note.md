# 딥 러닝을 이용한 자연어 처리 입문

- [자연어 처리란?](#자연어-처리란)
- [텍스트 전처리 (Text preprocessing)](#텍스트-전처리-text-preprocessing)
  * [토큰화 Tokenization](#토큰화-tokenization)
    + [1. 단어 토큰화 Word Tokenization](#1-단어-토큰화-word-tokenization)
    + [2. 토큰화 중 생기는 선택의 순간](#2-토큰화-중-생기는-선택의-순간)
    + [3. 토큰화에서 고려해야할 사항](#3-토큰화에서-고려해야할-사항)
    + [4. 문장 토큰화 Sentence Tokenization](#4-문장-토큰화-sentence-tokenization)
    + [5. 이진 분류기 Binary Classifier](#5-이진-분류기-binary-classifier)
    + [6. 한국어에서의 토큰화의 어려움.](#6-한국어에서의-토큰화의-어려움)
    + [7. 품사 태깅(Part-of-speech tagging)](#7-품사-태깅part-of-speech-tagging)
    + [8. NLTK와 KoNLPy를 이용한 영어, 한국어 토큰화 실습](#8-nltk와-konlpy를-이용한-영어-한국어-토큰화-실습)
  * [정제(Cleaning) and  정규화(Normalization)](#정제cleaning-and--정규화normalization)
    + [1. 규칙에 기반한 표기가 다른 단어들의 통합](#1-규칙에-기반한-표기가-다른-단어들의-통합)
    + [2. 대, 소문자 통합](#2-대-소문자-통합)
    + [3. 불필요한 단어의 제거 (Removing Unnecessary Words)](#3-불필요한-단어의-제거-removing-unnecessary-words)
    + [4. 정규 표현식(Regular Expression)](#4-정규-표현식regular-expression)
  * [어간 추출(Stemming) and 표제어 추출(Lemmatization)](#어간-추출stemming-and-표제어-추출lemmatization)
    + [1. 표제어 추출(Lemmatization)](#1-표제어-추출lemmatization)
    + [2. 어간 추출(Stemming)](#2-어간-추출stemming)
    + [3. 한국어에서의 어간 추출](#3-한국어에서의-어간-추출)
  * [불용어 (Stopword)](#불용어-stopword)
    + [1. NLTK에서 불용어 확인하기](#1-nltk에서-불용어-확인하기)
    + [2. NLTK를 통해서 불용어 제거하기](#2-nltk를-통해서-불용어-제거하기)
    + [3. 한국어에서 불용어 제거하기](#3-한국어에서-불용어-제거하기)
  * [정규 표현식(Regular Expression)](#정규-표현식regular-expression)
    + [1. 정규 표현식 문법과 모듈 함수](#1-정규-표현식-문법과-모듈-함수)
    + [2. 정규 표현식 실습](#2-정규-표현식-실습)
    + [3. 정규 표현식 모듈 함수 예제](#3-정규-표현식-모듈-함수-예제)
    + [4. 정규 표현식 텍스트 전처리 예제](#4-정규-표현식-텍스트-전처리-예제)
    + [5. 정규 표현식을 이용한 토큰화](#5-정규-표현식을-이용한-토큰화)
  * [정수 인코딩 (Integer Encoding)](#정수-인코딩-integer-encoding)
    + [1. 정수 인코딩 (Integer Encoding)](#1-정수-인코딩-integer-encoding)
    + [2. 케라스 (Keras)의 텍스트 전처리](#2-케라스-keras의-텍스트-전처리)
  * [패딩 (Padding)](#패딩-padding)
    + [1. Numpy 로 패딩하기](#1-numpy-로-패딩하기)
    + [2. 케라스 전처리 도구로 패딩](#2-케라스-전처리-도구로-패딩)
  * [원 - 핫 인코딩 (One-Hot Encoding)](#원---핫-인코딩-one-hot-encoding)
    + [1. 원 - 핫 인코딩 (One - Hot Encoding) 이란?](#1-원---핫-인코딩-one---hot-encoding-이란)
    + [2. 케라스 (Keras)를 이용한 원-핫 인코딩 (One-Hot-Encoding)](#2-케라스-keras를-이용한-원-핫-인코딩-one-hot-encoding)
    + [3. 원 - 핫 인코딩(One-Hot Encoding)의 한계](#3-원---핫-인코딩one-hot-encoding의-한계)
  * [데이터의 분리 (Splitting Data)](#데이터의-분리-splitting-data)
    + [1. 지도 학습 (Supervised Learning)](#1-지도-학습-supervised-learning)
    + [2. X와 y 분리하기](#2-x와-y-분리하기)
    + [3. 테스트 데이터 분리하기](#3-테스트-데이터-분리하기)
  * [한국어 전처리 패키지 (Text Preprocessing Tools for Korean Text)](#한국어-전처리-패키지-text-preprocessing-tools-for-korean-text)
    + [1. PyKoSpacing](#1-pykospacing)
    + [2. Py - Hanspell](#2-py---hanspell)
    + [3. SOYNLP 를 이용한 단어 토큰화](#3-soynlp-를-이용한-단어-토큰화)
    + [4. SOYNLP 를 이용한 반복되는 문자 정제](#4-soynlp-를-이용한-반복되는-문자-정제)
    + [5. Customized KoNLPy](#5-customized-konlpy)
- [언어 모델 (Language Model)](#언어-모델-language-model)
  * [언어 모델 (Language Model) 이란?](#언어-모델-language-model-이란)
    + [1. 언어 모델 (Language Model)](#1-언어-모델-language-model)
    + [2. 단어 시퀀스의 확률 할당](#2-단어-시퀀스의-확률-할당)
    + [3. 주어진 이전 단어들로부터 다음 단어 예측하기](#3-주어진-이전-단어들로부터-다음-단어-예측하기)
    + [4. 언어 모델의 간단한 직관](#4-언어-모델의-간단한-직관)
    + [5. 검색 엔진에서의 언어 모델의 예](#5-검색-엔진에서의-언어-모델의-예)
  * [통계적 언어 모델 (Statistical Language Model, SLM)](#통계적-언어-모델-statistical-language-model-slm)
    + [1. 조건부 확률](#1-조건부-확률)
    + [2. 문장에 대한 확률](#2-문장에 대한 확률)
    + [3. 카운트 기반의 접근](#3-카운트-기반의 접근)
    + [4. 카운트 기반 접근의 한계 - 희소 문제 (Sparsity Problem)](#4-카운트-기반-접근의-한계---희소-문제-sparsity-problem)
  * [N-gram 언어 모델 (N-gram Language Model)](#n-gram-언어-모델-ngram-language-model)
    + [1. Corpus 에서 카운트 하지 못하는 경우의 감소](#1-corpus-에서-카운트-하지-못하는-경우의-감소)
    + [2. N-gram](#2-n-gram)
    + [3. N-gram Language Model의 한계](#3-n-gram-language-modelㅢ-한계)
    + [4. 적용 분야(Domain)에 맞는 코퍼스의 수집](#4-적용-분야domain에-맞는-코퍼스의-수집)
    + [5. 인공 신경망을 이용한 언어 모델(Neural Network Based Language Model)](#5-인공-신경망을-이용한-언어-모델neural-network-based-language-model)
  * [한국어에서의 언어 모델 (Language Model for Korean Sentences)](#한국어에서의-언어-모델-language-model-for-korean-sentences)
    + [1. 한국어는 어순이 중요하지 않다.](#1-한국어는-어순이-중요하지-않다)
    + [2. 한국어는 교착어이다.](#2-한국어는-교착어이다)
    + [한국어는 띄어쓰기가 제대로 지켜지지 않는다.](#한국어는-띄어쓰기가-제대로-지켜지지-않는다)
  * [펄플렉서티 (Perplexity)](#펄플렉서티-perplexity)
    + [1. 언어 모델의 평가 방법 (Evaluation metric) : Perplexity 줄여서 PPL](#1-언어-모델의-평가-방법-evaluation-metric--perplexity-줄여서-ppl)
    + [2. 분기 계수(Branching factor)](#2-분기-계수branching-factor)
    + [3. 기존 언어 모델 vs 인공 신경망을 이용한 언어 모델](#3-기존-언어-모델-vs-인공-신경망을-이용한-언어-모델)
- [카운트 기반의 단어 표현 (Count based word Representation)](#카운트-기반의-단어-표현-count-based-word-representation)
  * [다양한 단어의 표현 방법](#다양한-단어의-표현-방법)
    + [1. 단어의 표현 방법](#1-단어의-표현-방법)
    + [2. 단어 표현의 카테고리화](#2-단어-표현의-카테고리화)
  * [Bag of Words(BoW)](#bag-of-wordsbow)
    + [2. Bag of Words 의 다른 예제들](#2-bag-of-words-의-다른-예제들)
    + [3. CounVectorizer 클래스로 BoW 만들기](#3-counvectorizer-클래스로-bow-만들기)
    + [4. 불용어를 제거한 BoW 만들기](#4-불용어를-제거한-bow-만들기)
  * [문서 단어 행렬 (Document - Term Matrix, DTM)](#문서-단어-행렬-document---term-matrix-dtm)
    + [1. 문서 단어 행렬 (Document - Term Matrix, DTM) 의 표기법](#1-문서-단어-행렬-document---term-matrix-dtm-의-표기법)
    + [2. 문서 단어 행렬의 한계](#2-문서-단어-행렬의-한계)
  * [TF-IDF (Term Frequency-Inverse Document Freqency)](#tfidf-term-frequency-inverse-document-freqency)
    + [1. TF - IDF (단어 빈도- 역 문서 빈도, Term Freqency - Inverse Document Frequency)](#1-tf---idf-단어-빈도--역-문서-빈도-term-freqency---inverse-document-frequency)
    + [2. 파이썬으로 TF-IDF 직접 구현하기](#2-파이썬으로-tf-idf-직접-구현하기)
    + [3. 사이킷런을 이용한 DTM과 TF-IDF 실습](#3-사이킷런을-이용한-dtm과-tf-idf-실습)
- [문서 유사도 (Document Similarity)](#문서-유사도-document-similarity)
  * [코사인 유사도 (Cosine Similarity)](#코사인-유사도-cosine-similarity)
    + [1. 코사인 유사도](#1-코사인-유사도)
    + [2. 유사도를 이용한 추천 시스템 구현하기](#2-유사도를-이용한-추천-시스템-구현하기)
  * [여러가지 유사도 기법](#여러가지-유사도-기법)
    + [1. 유클리드 거리 (Euclidean distance)](#1-유클리드-거리-euclidean-distance)
    + [2. 자카드 유사도 (Jaccard similarity)](#2-자카드-유사도-jaccard-similarity)
- [토픽 모델링 (Topic Modeling)](#토픽-모델링-topic-modeling)
  * [잠재 의미 분석 (Latent Semantic Analysis, LSA)](#잠재-의미-분석-latent-semantic-analysis-lsa)
    + [1. 특이값 분해 (Singular Value Decomposition, SVD)](#1-특이값-분해-singular-value-decomposition-svd)
    + [2. 절단된 SVD (Truncated SVD)](#2-절단된-svd-truncated-svd)
    + [3. 잠재 의미 분석 (Latent Semantic Analysis , LSA)](#3-잠재-의미-분석-latent-semantic-analysis--lsa)
    + [4. 실습을 통한 이해](#4-실습을-통한-이해)
    + [5.  LSA의 장단점 (Pros and Cons of LSA)](#5-lsa의-장단점-pros-and-cons-of-lsa)
  * [잠재 디리클레 할당 (Latent Dirichlet Allocation, LDA)](#-잠재-디리클래-할당-latent-dirichlet-allocation-lda)
    + [1. 잠재 디리클레 할당 (Latent Dirichlet Allocation, LDA) 개요](#1-잠재-디리클레-할당-latent-dirichlet-allocation-lda-개요)
    + [2. LDA의 가정](#2-lda의-가정)
    + [3. LDA 수행하기](#3-lda-수행하기)
    + [4. 잠재 디리클레 할당과 잠재 의미 분석의 차이](#4-잠재-디리클레-할당과-잠재-의미-분석의-차이)
    + [5. 실습](#5-실습)
    + [6. 실습 (gensim 사용)](#6-실습-gensim-사용)

# 자연어 처리란?

자연어(natural language)란 우리가 일상 생활에서 사용하는 언어 .

자연어 처리(natural language processing)란 이러한 자연어의 의미를 분석하여 컴퓨터가 처리할 수 있도록 하는 일.

음성 인식, 내용 요약, 번역, 사용자의 감성 분석, 텍스트 분류 작업(스팸 메일 분류, 뉴스 기사 카테고리 분류), 질의 응답 시스템, 챗봇과 같은 곳에서 사용되는 분야

---

# 텍스트 전처리 (Text preprocessing)

용도에 맞게 텍스트를 사전에 처리하는 작업.

자연어 처리에서 크롤링 등으로 얻어낸 코퍼스 데이터가 필요에 맞게 전처리되지 않은 상태라면???

토큰화(tokenization) & 정제(cleaning) & 정규화(normalization)하는 일을 하게 됩니다.

## 토큰화 Tokenization

주어진 코퍼스(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업

토큰의 단위가 상황에 따라 다르지만, 보통 의미있는 단위로 토큰을 정의

### 1. 단어 토큰화 Word Tokenization

토큰의 기준을 단어(word)로 하는 경우, 단어 토큰화(word tokenization)라고 합니다. 

다만, 여기서 단어(word)는 단어 단위 외에도 단어구, 의미를 갖는 문자열로도 간주되기도 합니다.

입력으로부터 구두점(punctuation)과 같은 문자는 제외시키는 간단한 단어 토큰화 작업
입력: Time is an illusion. Lunchtime double so!
출력 : "Time", "is", "an", "illusion", "Lunchtime", "double", "so"

보통 토큰화 작업은 단순히 구두점이나 특수문자를 전부 제거하는 정제(cleaning) 작업을 수행하는 것만으로 해결되지 않는다. 

구두점, 특수문자를 전부 제거하면 토큰이 의미를 잃어버리는 경우가 발생하기도 합니다. 심지어 한국어는 띄어쓰기만으로는 단어 토큰을 구분하기 어렵다. 

---

### 2. 토큰화 중 생기는 선택의 순간

```python
from nltk.tokenize import word_tokenize  
print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))  
['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

from nltk.tokenize import WordPunctTokenizer  
print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

from tensorflow.keras.preprocessing.text import text_to_word_sequence
print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
```

---

### 3. 토큰화에서 고려해야할 사항

1) 구두점이나 특수 문자를 단순 제외해서는 안 된다.

Ph.D  AT&T $45.55 01/02/06

2) 줄임말과 단어 내에 띄어쓰기가 있는 경우.

New York   rock 'n' roll   we're

3) 표준 토큰화 예제

표준으로 쓰이고 있는 토큰화 방법 중 하나인 Penn Treebank Tokenization의 규칙

규칙 1. 하이푼으로 구성된 단어는 하나로 유지한다.

규칙 2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.

```python
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))
['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
```

---

### 4. 문장 토큰화 Sentence Tokenization

갖고있는 코퍼스 내에서 문장 단위로 구분하는 작업으로 문장 분류(sentence segmentation)라고도 부른다.

주의) 마침표는 문장의 끝이 아니더라도 등장할 수 있다.

'IP 192.168.56.31 서버에 들어가서 로그 파일 저장해서 ukairia777@gmail.com로 결과 좀 보내줘. 그러고나서 점심 먹으러 가자.'

NLTK에서는 영어 문장의 토큰화를 수행하는 sent_tokenize를 지원하고 있다.

```python
from nltk.tokenize import sent_tokenize
text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print(sent_tokenize(text))
['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']

from nltk.tokenize import sent_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```

한국어에 대한 문장 토큰화 도구 또한 존재합니다. 

```python
import kss

text='딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?'
print(kss.split_sentences(text))
['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어려워요.', '농담아니에요.', '이제 해보면 알걸요?']
```

---

### 5. 이진 분류기 Binary Classifier

문장 토큰화에서의 예외 사항을 발생시키는 마침표의 처리를 위해서 입력에 따라 두 개의 클래스로 분류하는 이진 분류기(binary classifier)를 사용하기도 한다.

물론, 여기서 말하는 두 개의 클래스는

1. 마침표(.)가 단어의 일부분일 경우. 즉, 마침표가 약어(abbreivation)로 쓰이는 경우

2. 마침표(.)가 정말로 문장의 구분자(boundary)일 경우를 의미할 것입니다.

이러한 문장 토큰화를 수행하는 오픈 소스로는 NLTK, OpenNLP, 스탠포드 CoreNLP, splitta, LingPipe 등이 있습니다.

### 6. 한국어에서의 토큰화의 어려움.

한국어의 경우에는 띄어쓰기 단위가 되는 단위를 '어절'이라고 하는데 즉, 어절 토큰화는 한국어 NLP에서 지양되고 있다.

어절 토큰화와 단어 토큰화가 같지 않기 때문이다. 

그 이유는 한국어가 영어와는 다른 형태를 가지는 언어인 교착어라는 점에서 기인합니다. 교착어란 조사, 어미 등을 붙여서 말을 만드는 언어를 말합니다.

한국어 토큰화에서는 형태소(morpheme)란 개념을 반드시 이해해야 합니다.

1) 한국어는 교착어이다.

'그가', '그에게', '그를' , ... 

대부분의 한국어 NLP에서 조사는 분리해줄 필요가 있다.

한국어에서는 형태소 토큰화를 수행해야 한다.

2) 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않는다.

### 7. 품사 태깅(Part-of-speech tagging)

단어는 표기는 같지만, 품사에 따라서 단어의 의미가 달라지기도 한다.

ex) '못' : nail, not

품사 태깅 : 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는 지를 구분하는 작업

### 8. NLTK와 KoNLPy를 이용한 영어, 한국어 토큰화 실습

NLTK 에서 영어 코퍼스에 품사 태깅 기능을 지원하고 있다. 

```python
from nltk.tokenize import word_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
print(word_tokenize(text))

['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']

from nltk.tag import pos_tag
x=word_tokenize(text)
pos_tag(x)

[('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
```

한국어 자연어 처리를 위해서는 KoNLPy 패키지 사용.

KoNLPy 를 통해서 사용할 수 있는 형태소 분석기로 Okt, Mecab, Komoran, Hannanum, Kkma 가 있다. 

Okt 사용

```python
from konlpy.tag import Okt  
okt=Okt()  
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']

print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

[('열심히','Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]

print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

['코딩', '당신', '연휴', '여행']
```

Kkma 사용

```python
from konlpy.tag import Kkma  
kkma=Kkma()  
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']

print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

[('열심히','MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]

print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

['코딩', '당신', '연휴', '여행']
```

---

## 정제(Cleaning) and  정규화(Normalization)

**토큰화(tokenization)** : 코퍼스에서 용도에 맞게 토큰을 분류하는 작업

토큰화 작업 전, 후에 데이터를 용도에 맞게 정제 및 정규화

**정제(Cleaning)** : 갖고 있는 코퍼스로부터 노이즈 데이터 제거

**정규화(Normalization)** : 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만들어줌

### 1. 규칙에 기반한 표기가 다른 단어들의 통합

같은 의미를 갖고 있음에도, 표기가 다른 단어들을 하나의 단어로 정규화 하는 방법. 

ex) USA, US    uh-huh, uhhuh 

다음 챕터에서 표기가 다른 단어들을 통합하는 방법인 

**어간 추출**(stemming), **표제어 추출(**lemmatization) 배울 예정

### 2. 대, 소문자 통합

무작정 통합해서는 안 된다!

💡US(미국) 과 us(우리) 

하지만 결국에는 예외 사항을 크게 고려하지 않고, 모든 코퍼스를 소문자로 바꾸는 것이 종종 더 실용적인 해결책이 된다.

### 3. 불필요한 단어의 제거 (Removing Unnecessary Words)

불필요 단어들을 제거 하는 방법

- 불용어
- 등장 빈도가 적은 단어 (Removing Rare Words)
- 길이가 짧은 단어 (Removing words with very a short length)

영어 단어의 평균 길이는 6-7 / 한국어 단어의 평균 길이 2-3 로 추정

ex) dragon 용 / school 학교 

이러한 특성으로 인해 영어는 길이가 2-3 이하인 단어를 제거하는 것 만으로도 크게 의미를 갖지 못하는 단어를 줄이는 효과를 갖고 있다. 

```python
import re
text = "I was wondering if anyone out there could enlighten me on this car."
shortword = re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('', text))

was wondering anyone out there could enlighten this car.
```

### 4. 정규 표현식(Regular Expression)

얻어낸 코퍼스에서 노이즈 데이터의 특징을 잡아낼 수 있다면, **정규 표현식**을 통해서 이를 제거할 수 있는 경우도 많다. 

---

## 어간 추출(Stemming) and 표제어 추출(Lemmatization)

자연어 처리에서 전처리, 더 정확히는 정규화의 지향점은 언제나 갖고 있는 코퍼스로부터 **복잡성을 줄이는 일**

단어의 빈도수를 기반으로 문제를 풀고자 하는 **BoW(Bag of Words)**표현을 사용하는 자연어 처리 문제에서 주로 사용됨.

**표제어 추출**과 **어간 추출**의 **차이** 
표제어 추출 
문맥을 고려하며, 수행했을 때의 결과는 해당 단어의 품사 정보를 보존.(POS 태그를 보존한다고 생각하면 됨)

어간 추출
품사 정보가 보존되지 않음. (POS 태그 고려 X)
어간 추출 한 결과가 사전에 존재하지 않는 단어일 경우가 많다.

### 1. 표제어 추출(Lemmatization)

ex) am, are, is 의 뿌리 단어 be  → 이 단어들의 표제어 be

표제어 추출을 하는 가장 섬세한 방법

단어의 **형태학적 파싱**을 먼저 진행

형태소는 두 가지 종류가 있다. 어간(stem)과 접사(affix)
1) 어간 stem
: 단어의 의미를 담고 있는 단어의 핵심 부분

2)접사 affix
: 단어에 추가적인 의미를 주는 부분

**형태학적 파싱**이란?

이 두 가지 구성 요소를 분리하는 작업

ex) cats → cat(어간) + -s(접사)

```python
from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([n.lemmatize(w) for w in words])

['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 
'fly', **'dy'**, 'watched', **'ha'**, 'starting']
```

'dy' 나 'ha' 와 같이 의미를 알 수 없는 적절하지 못한 단어 출력

lemmatizer 가 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있기 때문이다. 

WordNetLemmatizer 는 입력으로 단어가 동사 품사라는 사실을 알려줄 수 있음.

```python
n.lemmatize('dies', 'v')

'die'

n.lemmatize('has', 'v')

'have'
```

### 2. 어간 추출(Stemming)

형태학적 분석을 **단순화한 버젼**이라고 볼 수도 있고, 
**정해진 규칙**만 보고 단어의 어미를 자르는 어림짐작의 작업이라 볼 수 도 있다.

즉, **섬세한 작업이 아니기 때문에** 결과 단어가 사전에 존재하지 않는 단어일 수 있다.

ex) 어간 추출 알고리즘 중 하나인 포터 알고리즘(Porter Algorithm)

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s = PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words=word_tokenize(text)
print(words)

['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']

print([s.stem(w) for w in words])

['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']
```

```python
# 가령, 포터 알고리즘의 어간 추출은 이러한 규칙을 가짐.
# ALIZE → AL
# ANCE → 제거
# ICAL → IC

words=['formalize', 'allowance', 'electricical']
print([s.stem(w) for w in words])

['formal', 'allow', 'electric']
```

같은 단어에 대해서 표제어 추출과 어간 추출의 결과 차이

**Stemming** 

am → am

the going → the go

having → hav

**Lemmatization**

am → be

the going → the going

having → have

### 3. 한국어에서의 어간 추출

한국어는 5언 9품사의 구조를 가짐

그 중에 **용언**에 해당되는 '**동사**'와 '**형용사**'는 

**어간**(stem)과 **어미**(ending)의 결합으로 구성됨. 

**(1) 활용 (conjugation)**

활용이란?     용언의 어간과 어미를 가지는 일

활용은 어간이 어미를 취할 때, 

어간의 모습이 일정하다면 **규칙 활용**

ex) 잡/어간 + 다/어미

규칙 기반으로 어미를 단순히 분리해주면 어간 추출이 됨.

어간이나 어미의 모습이 변하는 **불규칙 활용**

ex) '듣-'  →  '듣/들-'  

단순한 분리 만으로 어간 추출이 되지 않고 좀 더 복잡한 규칙을 필요로 함.

---

## 불용어 (Stopword)

자주 등장하지만 분석을 하는 것에 있어서는 큰 도움이 되지 않는 단어

ex) I, my, me

NLTK에서는 위와 같은 100여개 이상의 영어 단어들을 불용어로 패키지 내에서 미리 정의하고 있다.

### 1. NLTK에서 불용어 확인하기

```python
from nltk.corpus import stopwords  
stopwords.words('english')[:10]

['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']
```

### 2. **NLTK를 통해서 불용어 제거하기**

```python
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(example)

result = []
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 

print(word_tokens) 
print(result)

['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
```

### 3. **한국어에서 불용어 제거하기**

간단하게는 토큰화 후에 조사, 접속사 등을 제거하는 방법이 있다.

하지만, 명사, 형용사와 같은 단어들 중에서도 불용어로서 제거하고 싶은 단어들이 생기기도 한다. 

결국에 사용자가 직접 불용어 사전을 만들게 되는 경우가 많다. 

```python
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"
# 위의 불용어는 명사가 아닌 단어 중에서 저자가 임의로 선정한 것으로 실제 의미있는 선정 기준이 아님
stop_words=stop_words.split(' ')
word_tokens = word_tokenize(example)

result = [] 
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 
# 위의 4줄은 아래의 한 줄로 대체 가능
# result=[word for word in word_tokens if not word in stop_words]

print(word_tokens) 
print(result)

['고기를', '아무렇게나', '구우려고', '하면', '안', '돼', '.', '고기라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']
['고기를', '구우려고', '안', '돼', '.', '고기라고', '다', '같은', '게', '.', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']
```

한국어 불용어를 제거하는 더 좋은 방법은 
코드 내에서 직접 정의하지 않고 **txt 파일이나 csv 파일로 수많은 불용어를 정리해놓고**, 이를 불러와서 사용하는 방법이 있다. 

---

## 정규 표현식(Regular Expression)

정규 표현식 모듈 re 를 이용하여 특정 규칙이 있는 텍스트 데이터를 정제.

### 1. 정규 표현식 문법과 모듈 함수

**1) 정규 표현식 문법**

<img src = "/image/Untitled.png" width = "600px">

**2) 정규표현식 모듈 함수**

<img src = "/image/Untitled 1.png" width = "600px">

### 2. 정규 표현식 실습

- 예제

    ```python
    import re
    r = re.compile('a.c')
    r.search('kkk') # 아무런 결과도 출력되지 않는다.

    r.search('kkkkabc')
    <re.Match object; span=(4, 7), match='abc'>
    ```

    ```python
    import re
    r=re.compile("ab?c")
    r.search("abbc") # 아무런 결과도 출력되지 않는다.

    r.search("abc")
    <_sre.SRE_Match object; span=(0, 3), match='abc'>  
    b가 있는 것으로 판단하여 abc를 매치하는 것을 볼 수 있습니다.

    r.search("ac")
    <_sre.SRE_Match object; span=(0, 2), match='ac'>
    ```

    ```python
    import re
    r=re.compile("ab*c")
    r.search("a") # 아무런 결과도 출력되지 않는다.

    r.search("ac")
    <_sre.SRE_Match object; span=(0, 2), match='ac'>  

    r.search("abc") 
    <_sre.SRE_Match object; span=(0, 3), match='abc'> 

    r.search("abbbbc") 
    <_sre.SRE_Match object; span=(0, 6), match='abbbbc'>
    ```

    ```python
    import re
    r=re.compile("ab+c")
    r.search("ac") # 아무런 결과도 출력되지 않는다.

    r.search("abc") 
    <_sre.SRE_Match object; span=(0, 3), match='abc'>   

    r.search("abbbbc") 
    <_sre.SRE_Match object; span=(0, 6), match='abbbbc'>
    ```

    ```python
    import re
    r=re.compile("^a")
    r.search("bbc") # 아무런 결과도 출력되지 않는다.

    r.search("ab")                                                                                                    
    <_sre.SRE_Match object; span=(0, 1), match='a'>
    ```

    ```python
    import re
    r=re.compile("ab{2}c")
    r.search("ac") # 아무런 결과도 출력되지 않는다.

    r.search("abc") # 아무런 결과도 출력되지 않는다.

    r.search("abbc")
    <_sre.SRE_Match object; span=(0, 4), match='abbc'>

    r.search("abbbbbc") # 아무런 결과도 출력되지 않는다.
    ```

    ```python
    import re
    r=re.compile("ab{2,8}c")
    r.search("ac") # 아무런 결과도 출력되지 않는다.
    r.search("ac") # 아무런 결과도 출력되지 않는다.
    r.search("abc") # 아무런 결과도 출력되지 않는다.
    r.search("abbc")
    <_sre.SRE_Match object; span=(0, 4), match='abbc'>
    r.search("abbbbbbbbc")
    <_sre.SRE_Match object; span=(0, 10), match='abbbbbbbbc'>
    r.search("abbbbbbbbbc") # 아무런 결과도 출력되지 않는다.
    ```

    ```python
    import re
    r=re.compile("a{2,}bc")
    r.search("bc") # 아무런 결과도 출력되지 않는다.
    r.search("aa") # 아무런 결과도 출력되지 않는다.
    r.search("aabc")
    <_sre.SRE_Match object; span=(0, 4), match='aabc'>
    r.search("aaaaaaaabc")
    <_sre.SRE_Match object; span=(0, 10), match='aaaaaaaabc'>
    ```

    ```python
    import re
    r=re.compile("[abc]") # [abc]는 [a-c]와 같다.
    r.search("zzz") # 아무런 결과도 출력되지 않는다.
    r.search("a")
    <_sre.SRE_Match object; span=(0, 1), match='a'> 
    r.search("aaaaaaa")                                                                                               
    <_sre.SRE_Match object; span=(0, 1), match='a'> 
    r.search("baac")      
    <_sre.SRE_Match object; span=(0, 1), match='b'>

    import re
    r=re.compile("[a-z]")
    r.search("AAA") # 아무런 결과도 출력되지 않는다.
    r.search("aBC")
    <_sre.SRE_Match object; span=(0, 1), match='a'>
    r.search("111") # 아무런 결과도 출력되지 않는다.
    ```

    ```python
    import re
    r=re.compile("[^abc]")
    r.search("a") # 아무런 결과도 출력되지 않는다.
    r.search("ab") # 아무런 결과도 출력되지 않는다.
    r.search("b") # 아무런 결과도 출력되지 않는다.
    r.search("d")
    <_sre.SRE_Match object; span=(0, 1), match='d'> 
    r.search("1")                                                                                                
    <_sre.SRE_Match object; span=(0, 1), match='1'>
    ```

### 3. 정규 표현식 모듈 함수 예제

- 예제

    **(1) re.match() 와 re.search()의 차이**

    ```python
    import re
    r=re.compile("ab.")
    r.search("kkkabc")  
    <_sre.SRE_Match object; span=(3, 6), match='abc'>   

    r.match("kkkabc")  #아무런 결과도 출력되지 않는다.
    r.match("abckkk")  
    <_sre.SRE_Match object; span=(0, 3), match='abc'>
    ```

    **(2) re.split()**

    ```python
    import re
    text="사과+딸기+수박+메론+바나나"
    re.split("\+",text)
    ['사과', '딸기', '수박', '메론', '바나나']
    ```

    **(3) re.findall()**

    ```python
    import re
    text="이름 : 김철수
    전화번호 : 010 - 1234 - 1234
    나이 : 30
    성별 : 남"""  
    re.findall("\d+",text)

    ['010', '1234', '1234', '30']

    re.findall("\d+", "문자열입니다.")

    [] # 빈 리스트를 리턴한다.
    ```

    **(4) re.sub()**

    ```python
    import re
    text="Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."
    re.sub('[^a-zA-Z]',' ',text)

    'Regular expression   A regular expression  regex or regexp     sometimes called a rational expression        is  in theoretical computer science and formal language theory  a sequence of characters that define a search pattern '
    ```

### 4. 정규 표현식 텍스트 전처리 예제

```python
import re  

text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""  

re.split('\s+', text)  
['100', 'John', 'PROF', '101', 'James', 'STUD', '102', 'Mac', 'STUD']

re.findall('\d+',text)  
['100', '101', '102']

re.findall('[A-Z]',text)
['J', 'P', 'R', 'O', 'F', 'J', 'S', 'T', 'U', 'D', 'M', 'S', 'T', 'U', 'D']
# 이는 우리가 원하는 결과가 아닙니다. 
# 이 경우, 여러가지 방법이 있겠지만 대문자가 연속적으로 4번 등장하는 경우로 조건을 추가해봅시다.

re.findall('[A-Z]{4}',text)  
['PROF', 'STUD', 'STUD']
# 대문자로 구성된 문자열들을 제대로 가져오는 것을 볼 수 있습니다. 
# 이름의 경우에는 대문자와 소문자가 섞여있는 상황입니다. 이름에 대한 행의 값을 갖고오고 싶다면 처음에 대문자가 등장하고, 그 후에 소문자가 여러번 등장하는 경우에 매치하게 합니다.

re.findall('[A-Z][a-z]+',text)
['John', 'James', 'Mac'] 

import re
letters_only = re.sub('[^a-zA-Z]', ' ', text)
```

### 5. 정규 표현식을 이용한 토큰화

NLTK에서 정규 표현식을 사용해 단어 토큰화를 수행하는 **RegexpTokenizer** 지원

```python
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\w]+")
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']

import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\s]+", gaps=True)

print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']

gaps=true 는 해당 정규 포현식을 토큰으로 나누기 위한 기준으로 사용한다는 의미
gaps=true 기재하지 않으면 공백들만 나온다.
```

---

## 정수 인코딩 (Integer Encoding)

자연어 처리에서는 텍스트를 숫자로 바꾸는 여러가지 기법들이 있다.

본격적인 첫 단계로 각 단어를 고유한 정수에 mapping 시키는 전처리 작업이 필요할 때가 있다.

인덱스를 부여하는 방법은 랜덤으로 부여하기도 하지만, 

보통은 단어에 대한 빈도수를 기준으로 정렬한 뒤에 부여한다.

### 1. 정수 인코딩 (Integer Encoding)

단어를 빈도수 순으로 정렬한 단어 집합을 만들고, 빈도수가 높은 순서대로 정수를 부여.

**1) dictionary 사용하기**

```python
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

text = sent_tokenize(text)
print(text)
['A barber is a person.', 'a barber is good person.', 'a barber is huge person.', 'he Knew A Secret!', 'The Secret He Kept is huge secret.', 'Huge secret.', 'His barber kept his word.', 'a barber kept his word.', 'His barber kept his secret.', 'But keeping and keeping such a huge secret to himself was driving the barber crazy.', 'the barber went up a huge mountain.']

# 정제와 단어 토큰화
vocab = {} # 파이썬의 dictionary 자료형
sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence = word_tokenize(i) # 단어 토큰화를 수행합니다.
    result = []

    for word in sentence: 
        word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거합니다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거합니다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1
    sentences.append(result) 
print(sentences)

[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

print(vocab)
{'barber': 8, 'person': 3, 'good': 1, 'huge': 5, 'knew': 1, 'secret': 6, 'kept': 4, 'word': 2, 'keeping': 2, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1}

vocab_sorted = sorted(voca.items(), key = lambda x:x[1], reverse=True)

print(vocab_sorted)
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3), ('word', 2), ('keeping', 2), ('good', 1), ('knew', 1), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)]

#이제 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여합니다.

word_to_index = {}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외한다.
        i=i+1
        word_to_index[word] = i
print(word_to_index)
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7}
```

자연어 처리를 하다 보면, 빈도수가 가장 높은 n개의 단어만 사용하고 싶은 경우가 있다.

상위 n개의 단어만 사용하고 싶다고 하면 vocab에서 value가 1-n까지인 단어들만 사용하면 된다.

```python
vocab_size = 5
words_frequency = [w for w,c in word_to_index.items() if c >= vocab_size + 1] # 인덱스가 5 초과인 단어 제거
for w in words_frequency:
    del word_to_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
print(word_to_index)
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

word_to_index를 사용하여 단어 토큰화가 된 상태로 저장된 sentences에 있는 각 단어를 정수로 바꾸는 작업.

그런데 두 번째 문장인 ['barber', 'good', 'person']에는 더 이상 word_to_index에는 존재하지 않는 단어인 'good'이라는 단어가 있다.

이처럼 단어 집합에 존재하지 않는 단어들을 **Out-Of-Vocabulary(단어 집합에 없는 단어)** '**OOV**'. word_to_index에 'OOV'란 단어를 새롭게 추가하고, 단어 집합에 없는 단어들은 'OOV'의 인덱스로 인코딩.

```python
word_to_index['OOV'] = len(word_to_index) + 1

encoded = []
for s in sentences:
    temp = []
    for w in s:
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['OOV'])
    encoded.append(temp)
print(encoded)
[[1, 5], [1, 6, 5], [1, 3, 5], [6, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [6, 6, 3, 2, 6, 1, 6], [1, 6, 3, 6]]
```

이보다 좀 더 쉽게 하기 위해 Counter, FreqDist, enumerate 또는 keras 토크나이저를 사용하는 것을 권장.

**2) Counter 사용하기**

```python
print(sentences)
[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

words = sum(sentences, [])
# 위 작업은 words = np.hstack(sentences)로도 수행 가능.
print(words)
['barber', 'person', 'barber', 'good', 'person', 'barber', 'huge', 'person', 'knew', 'secret', 'secret', 'kept', 'huge', 'secret', 'huge', 'secret', 'barber', 'kept', 'word', 'barber', 'kept', 'word', 'barber', 'kept', 'secret', 'keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy', 'barber', 'went', 'huge', 'mountain']
```

Counter() 로 중복을 제거하고 단어의 빈도수를 기록

```python
from collections import Counter

vocab = Counter(words) # 파이썬의 Counter 모듈을 이용하면 단어의 모든 빈도를 쉽게 계산할 수 있습니다.
print(vocab)
Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})
```

```python
vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
vocab
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]

# 이제 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여합니다.

word_to_index = {}
i = 0
for (word, frequency) in vocab :
    i = i+1
    word_to_index[word] = i
print(word_to_index)
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

**3) NLTK의 FreqDist 사용하기**

위에서 사용한 Counter()랑 같은 방법으로 사용할 수 있다.

```python
from nltk import FreqDist
import numpy as np
# np.hstack으로 문장 구분을 제거하여 입력으로 사용 . ex) ['barber', 'person', 'barber', 'good' ... 중략 ...
vocab = FreqDist(np.hstack(sentences))

vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
vocab
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]

# 앞서 Counter()를 사용했을 때와 결과가 같습니다. 
# 이전 실습들과 마찬가지로 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여합니다. 
# 그런데 이번에는 enumerate()를 사용하여 좀 더 짧은 코드로 인덱스를 부여하겠습니다.

word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
print(word_to_index)
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

**4) enumerate 이해하기**

```python
test=['a', 'b', 'c', 'd', 'e']
for index, value in enumerate(test): # 입력의 순서대로 0부터 인덱스를 부여함.
  print("value : {}, index: {}".format(value, index))

value : a, index: 0
value : b, index: 1
value : c, index: 2
value : d, index: 3
value : e, index: 4
```

### 2. 케라스 (Keras)의 텍스트 전처리

때로는 정수 인코딩을 위해서 Keras의 전처리 도구인 **Tokenizer** 를 사용하기도 한다.

```python
sentences=[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 
'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], 
['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], 
['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 
'went', 'huge', 'mountain']]

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) 
# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성한다.

print(tokenizer.word_index)
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
```

각 단어의 빈도수가 높은 순서대로 인덱스가 부여된 것을 확인할 수 있다.

```python
# 각 단어가 카운트를 수행하였을 때 몇 개였는지를 보고자 한다면 word_counts를 사용합니다.
print(tokenizer.word_counts)
OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
```

**texts_to_sequences()** 는 입력으로 들어온 코퍼스에 대해 각 단어를 이미 정해진 인덱스로 변환

```python
print(tokenizer.texts_to_sequences(sentences))
[[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
```

**tokenizer = Tokenizer(num_words=숫자)** 으로 빈도수가 높은 상위 몇 개의 단어만 사용하겠다고 지정할 수 있다.

```python
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개 단어만 사용
tokenizer.fit_on_texts(sentences)

```

num_words에서 +1을 더해서 값을 넣어주는 이유!
num_words는 숫자를 0부터 카운트. 
만약 5를 넣으면 0 ~ 4번 단어 보존을 의미! → 1번 단어부터 4번 단어만 남게됨
그렇기 때문에 1 ~ 5번 단어까지 사용하고 싶다면 num_words에 5+1인 값을 넣어줘야 한다.

실질적으로 숫자 0에 지정된 단어가 존재하지 않는데도 케라스 토크나이저가 숫자 0까지 단어 집합의 크기로 산정하는 이유는 자연어 처리에서 **패딩(padding)**이라는 작업 때문

```python
print(tokenizer.texts_to_sequences(sentences))
[[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
```

1번 단어부터 5번 단어까지만 보존되고 나머지 단어들은 제거된 것을 볼 수 있다.

Keras Tokenizer 는 기본적으로 단어 집합에 없는 단어인 OOV에 대해서는 단어를 정수로 바꾸는 과정에서 아예 단어를 제거한다는 특징이 있다. 

단어 집합에 없는 단어들은 OOV로 간주하여 보존하고 싶다면 Tokenizer의 인자 **oov_token**을 사용

```python
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')
# 빈도수 상위 5개 단어만 사용. 숫자 0과 OOV를 고려해서 단어 집합의 크기는 +2
tokenizer.fit_on_texts(sentences)

print('단어 OOV의 인덱스 : {}'.format(tokenizer.word_index['OOV']))
단어 OOV의 인덱스 : 1

print(tokenizer.texts_to_sequences(sentences))
[[2, 6], [2, 1, 6], [2, 4, 6], [1, 3], [3, 5, 4, 3], [4, 3], [2, 5, 1], [2, 5, 1], [2, 5, 3], [1, 1, 4, 3, 1, 2, 1], [2, 1, 4, 1]]
# 그 외 단어 집합에 없는 'good'과 같은 단어들은 전부 'OOV'의 인덱스인 1로 인코딩되었다.

```

---

## 패딩 (Padding)

각 문장(또는 문서)의 길이가 다를 수 있다. 

병렬 연산을 위해 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업

### 1. Numpy 로 패딩하기

```python
sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성한다.

encoded = tokenizer.texts_to_sequences(sentences)
print(encoded)
[[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
```

이제 모두 동일한 길이로 맞춰주기 위해 이 중에서 가장 길이가 긴 문장의 길이를 계산

```python
max_len = max(len(item) for item in encoded)
print(max_len)
7
```

이제 모든 문장의 길이를 7로!

이때 가상의 단어 'PAD'를 사용

'PAD'라는 단어가 있다고 가정하고, 이 단어는 0번 단어라고 정의

길이가 7보다 짧은 문장에는 숫자 0을 채워서 전부 길이 7로!

```python
for item in encoded: # 각 문장에 대해서
    while len(item) < max_len:   # max_len보다 작으면
        item.append(0)

padded_np = np.array(encoded)
padded_np

array([[ 1,  5,  0,  0,  0,  0,  0],
       [ 1,  8,  5,  0,  0,  0,  0],
       [ 1,  3,  5,  0,  0,  0,  0],
       [ 9,  2,  0,  0,  0,  0,  0],
       [ 2,  4,  3,  2,  0,  0,  0],
       [ 3,  2,  0,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  2,  0,  0,  0,  0],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 1, 12,  3, 13,  0,  0,  0]])
```

**패딩 (Padding) ?** 
데이터에 특정 값을 채워서 데이터의 크기(shape)를 조정하는 작업

숫자 0을 사용하고 있다면 **제로 패딩(zero padding)**

### 2. 케라스 전처리 도구로 패딩

Keras 에서 패딩을 위한 도구 **pad_sequences()** 제공

```python
encoded = tokenizer.texts_to_sequences(sentences)
print(encoded)

[[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]

padded = pad_sequences(encoded)
padded
array([[ 0,  0,  0,  0,  0,  1,  5],
       [ 0,  0,  0,  0,  1,  8,  5],
       [ 0,  0,  0,  0,  1,  3,  5],
       [ 0,  0,  0,  0,  0,  9,  2],
       [ 0,  0,  0,  2,  4,  3,  2],
       [ 0,  0,  0,  0,  0,  3,  2],
       [ 0,  0,  0,  0,  1,  4,  6],
       [ 0,  0,  0,  0,  1,  4,  6],
       [ 0,  0,  0,  0,  1,  4,  2],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 0,  0,  0,  1, 12,  3, 13]], dtype=int32)
```

뒤에 0을 채우고 싶다면 인자로 **padding='post'**.

```python
padded = pad_sequences(encoded, padding = 'post')
padded
array([[ 1,  5,  0,  0,  0,  0,  0],
       [ 1,  8,  5,  0,  0,  0,  0],
       [ 1,  3,  5,  0,  0,  0,  0],
       [ 9,  2,  0,  0,  0,  0,  0],
       [ 2,  4,  3,  2,  0,  0,  0],
       [ 3,  2,  0,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  2,  0,  0,  0,  0],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 1, 12,  3, 13,  0,  0,  0]], dtype=int32)
```

**max_len**의 인자로 정수를 주면, 해당 정수로 모든 문서의 길이를 동일하게 한다.

```python
padded = pad_sequences(encoded, padding = 'post', maxlen = 5)
padded
array([[ 1,  5,  0,  0,  0],
       [ 1,  8,  5,  0,  0],
       [ 1,  3,  5,  0,  0],
       [ 9,  2,  0,  0,  0],
       [ 2,  4,  3,  2,  0],
       [ 3,  2,  0,  0,  0],
       [ 1,  4,  6,  0,  0],
       [ 1,  4,  6,  0,  0],
       [ 1,  4,  2,  0,  0],
       [ 3,  2, 10,  1, 11],
       [ 1, 12,  3, 13,  0]], dtype=int32)
```

만약, 숫자 0이 아니라 다른 숫자를 패딩을 위한 숫자로 사용하고 싶다면 이 또한 가능!
pad_sequences의 인자로 **value**를 사용하면 0이 아닌 다른 숫자로 패딩이 가능합니다.

```python
# 현재 사용된 정수들과 겹치지 않도록, 단어 집합의 크기에 +1을 한 숫자로 사용
last_value = len(tokenizer.word_index) + 1 # 단어 집합의 크기보다 1 큰 숫자를 사용
print(last_value)
14

padded = pad_sequences(encoded, padding = 'post', value = last_value)
padded
array([[ 1,  5, 14, 14, 14, 14, 14],
       [ 1,  8,  5, 14, 14, 14, 14],
       [ 1,  3,  5, 14, 14, 14, 14],
       [ 9,  2, 14, 14, 14, 14, 14],
       [ 2,  4,  3,  2, 14, 14, 14],
       [ 3,  2, 14, 14, 14, 14, 14],
       [ 1,  4,  6, 14, 14, 14, 14],
       [ 1,  4,  6, 14, 14, 14, 14],
       [ 1,  4,  2, 14, 14, 14, 14],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 1, 12,  3, 13, 14, 14, 14]], dtype=int32)

```

---

## 원 - 핫 인코딩 (One-Hot Encoding)

컴퓨터는 문자보다는 숫자를 더 잘 처리.

자연어 처리에서는 문자를 숫자로 바꾸는 여러가지 기법들이 있다.

One-Hot Encoding 은 그 많은 기법 중에서 가장 기본적인 표현 방법.

One - Hot Encoding 을 위해서 먼저 해야 할 일 ! → 단어 집합을 만드는 일

**단어 집합(vocabulary) :** 서로 다른 단어들의 집합

book , books  ← 다른 단어

단어 집합의 단어들로 문자를 숫자(더 구체적으로는 **벡터**)로 바꾼다.

### 1. 원 - 핫 인코딩 (One - Hot Encoding) 이란?

 

단어 집합의 크기를 벡터의 차원으로, 

표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 

다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식. 

```python
from konlpy.tag import Okt  
okt=Okt()  
token=okt.morphs("나는 자연어 처리를 배운다")  
print(token)
['나', '는', '자연어', '처리', '를', '배운다']

word2index={}
for voca in token:
     if voca not in word2index.keys():
       word2index[voca]=len(word2index)
print(word2index)
{'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}

def one_hot_encoding(word, word2index):
       one_hot_vector = [0]*(len(word2index))
       index=word2index[word]
       one_hot_vector[index]=1
       return one_hot_vector

one_hot_encoding("자연어",word2index)
[0, 0, 1, 0, 0, 0]
```

### 2. 케라스 (Keras)를 이용한 원-핫 인코딩 (One-Hot-Encoding)

케라스는 원-핫 인코딩을 수행하는 유용한 도구 **to_categorical()**를 지원

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text="나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

t = Tokenizer()
t.fit_on_texts([text])
print(t.word_index) # 각 단어에 대한 인코딩 결과 출력.
{'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}

sub_text="점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded=t.texts_to_sequences([sub_text])[0]
print(encoded)
[2, 5, 1, 6, 3, 7]

one_hot = to_categorical(encoded)
print(one_hot)
[[0. 0. 1. 0. 0. 0. 0. 0.] #인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] #인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] #인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] #인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] #인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] #인덱스 7의 원-핫 벡터
```

### 3. 원 - 핫 인코딩(One-Hot Encoding)의 한계

원 핫 벡터는 **단어 집합의 크기**가 곧 **벡터의 차원 수**가 된다.

ex) 1,000개인 코퍼스를 가지고 원 핫 벡터를 만들면, 모든 단어 각각은 모두 1,000개의 차원을 가진 벡터가 된다.

원-핫 벡터는 단어의 **유사도를 표현하지 못한다**는 단점이 있다.

ex) 늑대, 호랑이, 강아지, 고양이라는 4개의 단어에 대해서 원-핫 인코딩을 해서 각각, [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]. 이 때 강아지와 늑대가 유사하고, 호랑이와 고양이가 유사하다는 것을 표현할 수 없다.

단어 간 유사성을 알 수 없다는 단점은 **검색 시스템** 등에서 심각한 문제.

단점을 해결하기 위해 단어의 잠재 의미를 반영하여 다차원 공간에 벡터화 하는 기법으로 크게 두 가지가 있다.

카운트 기반의 벡터화 방법:      LSA, HAL 등

예측 기반으로 벡터화 방법:      NNLM, RNNLM, Word2Vec, FastText 등

카운트 기반과 예측 기반 두 가지 방법을 모두 사용하는 방법:    GloVe

---

## 데이터의 분리 (Splitting Data)

지도 학습을 위한 데이터 분리 작업

### 1. 지도 학습 (Supervised Learning)

지도 학습의 훈련 데이터는 

정답이 무엇인지 맞춰야 하는 '**문제**'에 해당되는 데이터와 

레이블이라고 부르는 '**정답**'이 적혀있는 데이터로 구성되어 있다.

기계는 정답이 적혀져 있는 문제지를 문제와 정답을 함께 보면서 열심히 공부하고, 향후에 정답이 없는 문제에 대해서도 정답을 잘 예측해야 한다.

**<훈련 데이터>**

X_train : 문제지 데이터

y_train : 문제지에 대한 정답 데이터

**<테스트 데이터>**

X_test : 시험지 데이터

y_test : 시험지에 대한 정답 데이터

### 2. X와 y 분리하기

**1) zip 함수를 이용하여 분리하기**

zip()    :    동일한 개수를 가지는 시퀀스 자료형에서 각 순서에 등장하는

               원소들끼리 묶어주는 역할

```python
X,y = zip(['a', 1], ['b', 2], ['c', 3])
print(X)
print(y)
('a', 'b', 'c')
(1, 2, 3)

sequences=[['a', 1], ['b', 2], ['c', 3]] # 리스트의 리스트 또는 행렬 또는 뒤에서 배울 개념인 2D 텐서.
X,y = zip(*sequences) # *를 추가
print(X)
print(y)
('a', 'b', 'c')
(1, 2, 3)
```

**2) 데이터 프레임을 이용하여 분리하기**

```python
import pandas as pd

values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
df

```

<img src = "/image/Untitled 2.png" width = "600px">

```python
X=df['메일 본문']
y=df['스팸 메일 유무']

print(X)
0          당신에게 드리는 마지막 혜택!
1      내일 뵐 수 있을지 확인 부탁드...
2      도연씨. 잘 지내시죠? 오랜만입...
3    (광고) AI로 주가를 예측할 수 있다!
Name: 메일 본문, dtype: object

print(y)
0    1
1    0
2    0
3    1
Name: 스팸 메일 유무, dtype: int64

```

**3) Numpy를 이용하여 분리하기**

```python
import numpy as np
ar = np.arange(0,16).reshape((4,4))
print(ar)
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
X=ar[:, :3]
print(X)
[[ 0  1  2]
 [ 4  5  6]
 [ 8  9 10]
 [12 13 14]]
y=ar[:,3]
print(y)
[ 3  7 11 15]
```

### 3. 테스트 데이터 분리하기

X와 y가 분리된 데이터에 대해서 테스트 데이터를 분리하는 과정

**1) 사이킷 런을 이용하여 분리하기**

```python
import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
# 실습을 위해 임의로 X와 y가 이미 분리 된 데이터를 생성
print(X)
print(list(y)) #레이블 데이터
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
[0, 1, 2, 3, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
#3분의 1만 test 데이터로 지정.
#random_state 지정으로 인해 순서가 섞인 채로 훈련 데이터와 테스트 데이터가 나눠진다.
print(X_train)
print(X_test)
[[2 3]
 [4 5]
 [6 7]]
[[8 9]
 [0 1]]
print(y_train)
print(y_test)
[1, 2, 3]
[4, 0]
```

**2) 수동으로 분리하기**

```python
import numpy as np
X, y = np.arange(0,24).reshape((12,2)), range(12)
# 실습을 위해 임의로 X와 y가 이미 분리 된 데이터를 생성
print(X)
[[ 0  1]
 [ 2  3]
 [ 4  5]
 [ 6  7]
 [ 8  9]
 [10 11]
 [12 13]
 [14 15]
 [16 17]
 [18 19]
 [20 21]
 [22 23]]
print(list(y))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

n_of_train = int(len(X) * 0.8) # 데이터의 전체 길이의 80%에 해당하는 길이값을 구한다.
n_of_test = int(len(X) - n_of_train) # 전체 길이에서 80%에 해당하는 길이를 뺀다.
print(n_of_train)
print(n_of_test)
9
3

X_test = X[n_of_train:] #전체 데이터 중에서 20%만큼 뒤의 데이터 저장
y_test = y[n_of_train:] #전체 데이터 중에서 20%만큼 뒤의 데이터 저장
X_train = X[:n_of_train] #전체 데이터 중에서 80%만큼 앞의 데이터 저장
y_train = y[:n_of_train] #전체 데이터 중에서 80%만큼 앞의 데이터 저장

print(X_test)
print(list(y_test))

[[18 19]
 [20 21]
 [22 23]]
[9, 10, 11]
```

---

## 한국어 전처리 패키지 (Text Preprocessing Tools for Korean Text)

형태소와 문장 토크나이징 도구들인 KoNLPy와 KSS(Korean Sentence Splitter)와 함께 유용하게 사용할 수 있는 한국어 전처리 패키지들

### 1. PyKoSpacing

PyKoSpacing은 한국어 띄어쓰기 패키지로 띄어쓰기가 되어있지 않은 문장을 띄어쓰기를 한 문장으로 변환해주는 패키지

```python
!pip install git+https://github.com/haven-jeon/PyKoSpacing.git

sent = '김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.'

new_sent = sent.replace(" ", '') # 띄어쓰기가 없는 문장 임의로 만들기
print(new_sent)
김철수는극중두인격의사나이이광수역을맡았다.철수는한국유일의태권도전승자를가리는결전의날을앞두고10년간함께훈련한사형인유연재(김광수분)를찾으러속세로내려온인물이다.

from pykospacing import spacing

kospacing_sent = spacing(new_sent)
print(sent)
print(kospacing_sent)
김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.
김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.

```

### 2. Py - Hanspell

네이버 한글 맞춤법 검사기를 바탕으로 만들어진 패키지

```python
!pip install git+https://github.com/ssut/py-hanspell.git

from hanspell import spell_checker

sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
spelled_sent = spell_checker.check(sent)

hanspell_sent = spelled_sent.checked
print(hanspell_sent)
맞춤법 틀리면 왜 안돼? 쓰고 싶은 대로 쓰면 되지
```

이 패키지는 띄어쓰기 또한 보정한다. 

```python
# PyKoSpacing에 사용한 예제를 그대로 사용

spelled_sent = spell_checker.check(new_sent)

hanspell_sent = spelled_sent.checked
print(hanspell_sent)
print(kospacing_sent) # 앞서 사용한 kospacing 패키지에서 얻은 결과
김철수는 극 중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연제(김광수 분)를 찾으러 속세로 내려온 인물이다.
김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.

# PyKoSpacing과 결과가 거의 비슷하지만 조금 다르다.

```

### 3. SOYNLP 를 이용한 단어 토큰화

soynlp는 품사 태깅, 단어 토큰화 등을 지원하는 단어 토크나이저

SOYNLP ?
텍스트 데이터에서 특정 문자 시퀀스가 함께 **자주 등장하는 빈도가 높고**, 
앞 뒤로 조사 또는 완전히 다른 단어가 등장하는 것을 고려해
해당 문자 시퀀스를 형태소라고 판단하는 단어 토크나이저

**비지도 학습**으로 단어 토큰화를 한다는 특징

데이터에 **자주 등장하는 단어**들을 단어로 **분석**

soynlp 단어 토크나이저는 내부적으로 **단어 점수 표**로 동작

이 점수는 **응집 확률(cohesion probability)**과 **브랜칭 엔트로피(branching entropy)**를 활용

```python
!pip install soynlp
```

SOYNLP가 어떤 점에서 유용한지 정리

**1) 신조어 문제**

**기존의 형태소 분석기**는 **신조어**나 형태소 분석기에 **등록되지 않은 단어** 같은 경우에는 제대로 구분하지 못하는 단점

```python
from konlpy.tag import Okt
tokenizer = Okt()
print(tokenizer.morphs('에이비식스 이대휘 1월 최애돌 기부 요정'))
['에이', '비식스', '이대', '휘', '1월', '최애', '돌', '기부', '요정']
```

**2) 학습하기**

soynlp는 기본적으로 **학습에 기반**한 토크나이저이므로 학습에 필요한 한국어 문서를 다운로드

```python
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")
훈련 데이터를 다수의 문서로 분리합니다.

# 훈련 데이터를 다수의 문서로 분리
corpus = DoublespaceLineCorpus("2016-10-20.txt")
len(corpus)
30091
```

상위 3개의 문서만 출력

```python
i = 0
for document in corpus:
  if len(document) > 0:
    print(document)
    i = i+1
  if i == 3:
    break

19  1990  52 1 22
오패산터널 총격전 용의자 검거 서울 연합뉴스 경찰 관계자들이 19일 오후 서울 강북구 오패산 터널 인근에서 사제 총기를 발사해 경찰을 살해한 용의자 성모씨를 검거하고 있다 ... 중략 ... 숲에서 발견됐고 일부는 성씨가 소지한 가방 안에 있었다
테헤란 연합뉴스 강훈상 특파원 이용 승객수 기준 세계 최대 공항인 아랍에미리트 두바이국제공항은 19일 현지시간 이 공항을 이륙하는 모든 항공기의 탑승객은 삼성전자의 갤럭시노트7을 휴대하면 안 된다고 밝혔다 ... 중략 ... 이런 조치는 두바이국제공항 뿐 아니라 신공항인 두바이월드센터에도 적용된다  배터리 폭발문제로 회수된 갤럭시노트7 연합뉴스자료사진
```

soynlp는 학습 기반의 단어 토크나이저이므로 학습 과정을 거쳐야 한다. 

이는 전체 코퍼스로부터 **응집 확률**과 **브랜칭 엔트로피** 단어 점수표를 만드는 과정

 **WordExtractor.extract()**를 통해서 전체 코퍼스에 대해 단어 점수표를 계산

```python
word_extractor = WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()
training was done. used memory 5.186 Gb
all cohesion probabilities was computed. # words = 223348
all branching entropies was computed # words = 361598
all accessor variety was computed # words = 361598
```

**3) SOYLP의 응집 확률 (cohesion probability)**

내부 문자열(substring)이 얼마나 응집하여 자주 등장하는 지를 판단하는 척도

문자열을 문자 단위로 분리하여 내부 문자열을 만드는 과정에서 

왼쪽부터 순서대로 문자를 추가하면서 

각 문자열이 주어졌을 때 그 다음 문자가 나올 확률을 계산하여 누적곱을 한 값

이 값이 높을수록 전체 코퍼스에서 이 문자열 시퀀스는 하나의 단어로 등장할 가능성 높다

<img src = "/image/Untitled 3.png" width = "600px">

ex) '반포한강공원에'라는 7의 길이를 가진 문자 시퀀스에 대해서 각 내부 문자열의 스코어를 구하는 과정

<img src = "/image/Untitled 4.png" width = "600px">

```python
word_score_table["반포한"].cohesion_forward
0.08838002913645132
# 그렇다면 '반포한강'의 응집 확률은 '반포한'의 응집 확률보다 높을까요?

word_score_table["반포한강"].cohesion_forward
0.19841268168224552
# '반포한강'은 '반포한'보다 응집 확률이 높습니다. 그렇다면 '반포한강공'은 어떨까요?

word_score_table["반포한강공"].cohesion_forward
0.2972877884078849
# 역시나 '반포한강'보다 응집 확률이 높습니다. '반포한강공원'은 어떨까요?

word_score_table["반포한강공원"].cohesion_forward
0.37891487632839754
# '반포한강공'보다 응집 확률이 높습니다. 여기다가 조사 '에'를 붙인 '반포한강공원에'는 어떨까요?

word_score_table["반포한강공원에"].cohesion_forward
0.33492963377557666
# 오히려 '반포한강공원'보다 응집도가 낮아집니다.

# 결국 결합도는 '반포한강공원'일 때가 가장 높았다. 
# 응집도로 판단하기에 하나의 단어로 판단하기에 가장 적합한 문자열: '반포한강공원'
```

**4) SOYNLP의 브랜칭 엔트로피 (branching entropy)**

Branching Entropy는 확률 분포의 엔트로피 값을 사용

주어진 문자열에서 얼마나 다음 문자가 등장할 수 있는 지를 판단하는 척도

브랜칭 엔트로피의 값은 하나의 완성된 단어에 가까워질수록 

문맥으로 인해 점점 정확히 예측할 수 있게 되면서 점점 줄어드는 양상을 보인다.

```python
word_score_table["디스"].right_branching_entropy
1.6371694761537934

word_score_table["디스플"].right_branching_entropy
-0.0

# '디스' 다음에는 다양한 문자가 올 수 있으니까 1.63이라는 값을 가지는 반면, 
# '디스플'이라는 문자열 다음에는 다음 문자로 '레'가 오는 것이 너무나 명백하기 때문에
# 0이란 값을 가진다.

word_score_table["디스플레"].right_branching_entropy
-0.0
word_score_table["디스플레이"].right_branching_entropy
3.1400392861792916

# 갑자기 값이 증가합니다. 
# 문자 시퀀스 '디스플레이'라는 문자 시퀀스 다음에는 조사나 다른 단어와 같은 
# 다양한 경우가 있을 수 있기 때문
# 하나의 단어가 끝나면 그 경계 부분부터 다시 브랜칭 엔트로피 값이 증가하게 됨을 의미
```

**5) SOYNLP의 L tokenizer**

한국어는 띄어쓰기 단위로 나눈 어절 토큰은 주로 L 토큰 + R 토큰의 형식을 가질 때가 많다

L 토큰 + R 토큰으로 나누되, 분리 기준을 점수가 가장 높은 L 토큰을 찾아내는 원리

```python
from soynlp.tokenizer import LTokenizer

scores = {word:score.cohesion_forward for word, score in word_score_table.items()}
l_tokenizer = LTokenizer(scores=scores)
l_tokenizer.tokenize("국제사회와 우리의 노력들로 범죄를 척결하자", flatten=False)
[('국제사회', '와'), ('우리', '의'), ('노력', '들로'), ('범죄', '를'), ('척결', '하자')]
```

**6) 최대 점수 토크나이저**

띄어쓰기가 되지 않는 문장에서 점수가 높은 글자 시퀀스를 순차적으로 찾는 토크나이저

```python
from soynlp.tokenizer import MaxScoreTokenizer

maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
maxscore_tokenizer.tokenize("국제사회와우리의노력들로범죄를척결하자")
['국제사회', '와', '우리', '의', '노력', '들로', '범죄', '를', '척결', '하자']
```

### 4. SOYNLP 를 이용한 반복되는 문자 정제

SNS나 채팅 데이터와 같은 한국어 데이터의 경우

ㅋㅋ, ㅋㅋㅋ, ㅋㅋㅋㅋ와 같은 경우를 모두 서로 다른 단어로 처리하는 것은 불필요!

이에 반복되는 것은 하나로 정규화.

```python
from soynlp.normalizer import *
print(emoticon_normalize('앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠㅠㅠ', num_repeats=2))
아ㅋㅋ영화존잼쓰ㅠㅠ
아ㅋㅋ영화존잼쓰ㅠㅠ
아ㅋㅋ영화존잼쓰ㅠㅠ
아ㅋㅋ영화존잼쓰ㅠㅠ

# 의미없게 반복되는 것은 비단 이모티콘에 한정되지 않습니다.

print(repeat_normalize('와하하하하하하하하하핫', num_repeats=2))
print(repeat_normalize('와하하하하하하핫', num_repeats=2))
print(repeat_normalize('와하하하하핫', num_repeats=2))
와하하핫
와하하핫
와하하핫
```

### 5. Customized KoNLPy

형태소 분석기를 사용할 때, 이런 상황에 봉착한다면 어떻게?

```python
형태소 분석 입력 : '은경이는 사무실로 갔습니다.'
형태소 분석 결과 : ['은', '경이', '는', '사무실', '로', '갔습니다', '.']
```

형태소 분석기에 사용자 사전을 추가해줄 수 있다.

'은경이'는 하나의 단어이기 때문에 분리하지 말라고 형태소 분석기에 알려주는 것.

**Customized Konlpy** 는 사용자 사전 추가가 매우 쉬운 패키지

```python
pip install customized_konlpy

from ckonlpy.tag import Twitter
twitter = Twitter()
twitter.morphs('은경이는 사무실로 갔습니다.')
['은', '경이', '는', '사무실', '로', '갔습니다', '.']
```

형태소 분석기 Twitter에 add_dictionary('단어', '품사')와 같은 형식으로 사전 추가를 해줄 수 있다.

```python
twitter.add_dictionary('은경이', 'Noun')

twitter.morphs('은경이는 사무실로 갔습니다.')
['은경이', '는', '사무실', '로', '갔습니다', '.']
```

---

# 언어 모델 (Language Model)

단어 시퀀스(문장)에 확률을 할당하는 모델

언어 모델이 하는 일!           **이 문장은 적절해! 이 문장은 말이 안 돼!**  

통계에 기반한 전통적인 언어 모델(Statistical Languagel Model, **SLM**)

우리가 실제 사용하는 자연어를 근사하기에는 많은 한계

인공 신경망이 그러한 한계를 많이 해결

통계 기반 언어 모델은 많이 사용 용도가 줄었다.

통계 기반 언어 모델에서 배우는 **n-gram**은 자연어 처리 분야에서 활발하게 활용되고 있다.

## 언어 모델 (Language Model) 이란?

언어라는 현상을 모델링하고자 
단어 시퀀스(또는 문장)에 **확률을 할당(assign)**하는 모델

언어 모델을 만드는 방법

- 통계를 이용한 방법
- 인공 신경망을 이용한 방법

최근 자연어 처리의 신기술인 **GPT**나 **BERT** (인공 신경망 언어 모델의 개념을 사용) 인기 !

### 1. 언어 모델 (Language Model)

단어 시퀀스에 **확률을 할당(assign)**하는 일을 하는 모델

→ 가장 자연스러운 단어 시퀀스를 찾아내는 모델

단어 시퀀스에 확률을 할당하기 위해 가장 보편적으로 사용되는 방법

→  언어 모델이 이전 단어들이 주어졌을 때 다음 단어를 예측하도록 하는 것.

다른 유형의 언어 모델

주어진 양쪽의 단어들로부터 가운데 비어있는 단어를 예측하는 언어 모델

**언어 모델링(Language Modeling)**

 주어진 단어들로부터 아직 모르는 단어를 예측하는 작업

 언어 모델이 이전 단어들로부터 다음 단어를 예측하는 일

---

### 2. 단어 시퀀스의 확률 할당

자연어 처리에서 단어 시퀀스에 확률을 할당하는 일이 왜 필요할까?

**a. 기계 번역(Machine Translation):**

P(나는 버스를 탔다) > P(나는 버스를 태운다)

: 언어 모델은 두 문장을 비교하여 좌측의 문장의 확률이 더 높다고 판단.

**b. 오타 교정(Spell Correction)**

선생님이 교실로 부리나케P(달려갔다) > P(잘려갔다)

: 언어 모델은 두 문장을 비교하여 좌측의 문장의 확률이 더 높다고 판단.

**c. 음성 인식(Speech Recognition)**

P(나는 메롱을 먹는다) < P(나는 메론을 먹는다)

: 언어 모델은 두 문장을 비교하여 우측의 문장의 확률이 더 높다고 판단.

언어 모델은 위와 같이 **확률**을 통해 보다 **적절한 문장을 판단**

---

### 3. 주어진 이전 단어들로부터 다음 단어 예측하기

**A. 단어 시퀀스의 확률**

하나의 단어를 w, 단어 시퀀스을 대문자 W라고 한다면, 

n개의 단어가 등장하는 단어 시퀀스 W의 확률

$$P(W)=P(w1,w2,w3,w4,w5,...,wn)$$

**B. 다음 단어 등장 확률**

n-1개의 단어가 나열된 상태에서 n번째 단어의 확률

$$P(wn|w1,...,wn−1)$$

전체 단어 시퀀스 W의 확률

$$P(W)=P(w1,w2,w3,w4,w5,...wn)=∏P(wn|w1,...,wn−1)$$

---

### 4. 언어 모델의 간단한 직관

앞에 어떤 단어들이 나왔는지 고려하여 

후보가 될 수 있는 여러 단어들에 대해서 등장 **확률을 추정**하고

**가장 높은 확률**을 가진 단어를 선택

---

### 5. 검색 엔진에서의 언어 모델의 예

<img src = "/image/Untitled 5.png" width = "600px">

---

## 통계적 언어 모델 (Statistical Language Model, SLM)

### 1. 조건부 확률

$$p(B|A)=P(A,B)/P(A)$$

$$P(A,B)=P(A)P(B|A)$$

**조건부 확률의 연쇄 법칙(chain rule)**

$$P(x1,x2,x3...xn)=P(x1)P(x2|x1)P(x3|x1,x2)...P(xn|x1...xn−1)$$

---

### 2. 문장에 대한 확률

문장 '**An adorable little boy is spreading smiles**'의 확률

$$P(An adorable little boy is spreading smiles)=
P(An)×P(adorable|An)×P(little|An adorable)×P(boy|An adorable little)×P(is|An adorable little boy)P(An)×P(adorable|An)×P(little|An adorable)×P(boy|An adorable little)×P(is|An adorable little boy) ×P(spreading|An adorable little boy is)×P(smiles|An adorable little boy is spreading)$$

---

### 3. 카운트 기반의 접근

SLM은 이전 단어로부터 다음 단어에 대한 확률은 어떻게 구할까?

→ **카운트에 기반하여 확률을 계산.**

$$P\text{(is|An adorable little boy}) = \frac{\text{count(An adorable little boy is})}{\text{count(An adorable little boy })}$$

ex) 기계가 학습한 코퍼스 데이터에서 An adorable little boy가 100번 등장

그 다음에 is가 등장한 경우는 30번

 → 이 경우 $P(\text{is|An adorable little boy})$는 30%

---

### 4. 카운트 기반 접근의 한계 - 희소 문제 (Sparsity Problem)

기계에게 많은 코퍼스를 훈련시켜서 언어 모델을 통해 현실에서의 확률 분포를 근사하는 것이 언어 모델의 목표

그런데 카운트 기반으로 접근하려고 한다면 갖고 있는 코퍼스(corpus). 

즉, 다시 말해 기계가 훈련하는 데이터는 정말 **방대한 양**이 필요.

위와 같이 $P\text{(is|An adorable little boy})$를 구하는 경우

기계가 훈련한 코퍼스에 An adorable little boy is라는 단어 시퀀스가 없었다면 

이 단어 시퀀스에 대한 확률은 0.

또는 An adorable little boy라는 단어 시퀀스가 없었다면 

분모가 0이 되어 확률은 정의되지 않는다.

**희소 문제(sparsity problem)**

충분한 데이터를 관측하지 못하여 언어를 정확히 모델링하지 못하는 문제

위 문제를 완화하는 방법

n-gram, 스무딩, 백오프

희소 문제에 대한 근본적인 해결책은 X

이러한 한계로 인해 언어 모델의 트렌드는 인공 신경망 언어 모델로 넘어갔다.

---

## N-gram 언어 모델 (N-gram Language Model)

n-gram 언어 모델은 

이전에 등장한 모든 단어를 고려하는 것이 아니라 일부 단어만 고려하는 접근 방법을 사용

n의 의미?

일부 단어를 몇 개 보느냐를 결정

### 1. Corpus 에서 카운트 하지 못하는 경우의 감소

SLM의 한계

 훈련 코퍼스에 확률을 계산하고 싶은 문장이나 단어가 없을 수 있다는 점

 확률을 계산하고 싶은 문장이 길어질수록 코퍼스에서 그 문장이 존재하지 않을 가능성이  

 높다는 점

$$P(\text{is|An adorable little boy}) \approx\ P(\text{is|little boy})$$

이제는 단어의 확률을 구하고자 기준 단어의 앞 단어를 전부 포함해서 카운트하는 것이 아니라, 앞 단어 중 **임의의 개수만 포함**해서 카운트하여 근사

---

### 2. N-gram

갖고 있는 코퍼스에서 **n개의 단어 뭉치 단위**로 끊어서 이를 **하나의 토큰**으로 간주

**uni**grams : an, adorable, little, boy, is, spreading, smiles

**bi**grams : an adorable, adorable little, little boy, boy is, is spreading, spreading smiles

**tri**grams : an adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles

**4-**grams : an adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles

n-gram을 통한 언어 모델에서는 

다음에 나올 단어의 예측은 **오직 n-1개의 단어에만 의존**

ex) 4-gram 을 이용한 언어 모델

<img src = "/image/Untitled 6.png" width = "600px">

$$P(w\text{|boy is spreading}) = \frac{\text{count(boy is spreading}\ w)}{\text{count(boy is spreading)}}$$

---

### 3. N-gram Language Model의 한계

전체 문장을 고려한 언어 모델보다는 정확도가 떨어질 수밖에 없다.

**1) 희소 문제 (Sparsity Problem)** 

n-gram 언어 모델도 여전히 n-gram에 대한 희소 문제가 존재

**2) n을 선택하는 것은 trade-off 문제**

n을 크게 선택하면 

훈련 코퍼스에서 해당 n-gram을 카운트할 수 있는 확률은 적어짐 → 희소 문제는 점점 심각

n을 작게 선택하면 

훈련 코퍼스에서 카운트는 잘 되겠지만 근사의 정확도는 현실의 확률분포와 멀어짐

정확도를 높이려면 **n은 최대 5를 넘게 잡아서는 안 된다고 권장**

---

### 4. 적용 분야(Domain)에 맞는 코퍼스의 수집

어떤 분야인지, 어떤 어플리케이션인지에 따라서 특정 단어들의 확률 분포는 다르다.

언어 모델에 사용하는 코퍼스를 해당 도메인의 코퍼스를 사용한다면 

당연히 언어 모델이 제대로 된 언어 생성을 할 가능성이 높아진다.

---

### 5. 인공 신경망을 이용한 언어 모델(Neural Network Based Language Model)

n-gram 언어 모델의 한계를 극복하기 위해 

분모, 분자에 숫자를 더해 카운트 했을 때 확률이 0을 방지하는 등의 여러 일반화 방법 존재

그럼에도 본질적으로 n-gram 언어 모델에 대한 취약점을 완전히 해결하지는 못함

인공 신경망을 이용한 언어 모델이 많이 사용되고 있다.

---

## 한국어에서의 언어 모델 (Language Model for Korean Sentences)

한국어 자연어 처리는 영어보다 훨씬 어렵다.

### 1. 한국어는 어순이 중요하지 않다.

ex)

① 나는 운동을 합니다 체육관에서.

② 나는 체육관에서 운동을 합니다.

③ 체육관에서 운동을 합니다.

④ 나는 운동을 체육관에서 합니다.

단어 순서를 뒤죽박죽으로 바꾸어도 한국어는 의미가 전달 되기 때문에 

확률에 기반한 언어 모델이 제대로 다음 단어를 예측하기가 어렵다.

---

### 2. 한국어는 교착어이다.

대표적인 예로 교착어인 한국어에는 조사가 있다.

**띄어쓰기 단위인 어절 단위**로 토큰화를 할 경우

문장에서 발생 가능한 단어의 수가 굉장히 늘어난다.

'그녀'라는 단어 하나만 해도 

그녀가, 그녀를, 그녀의, 그녀와, 그녀로, 그녀께서, 그녀처럼 등과 같이 다양한 경우가 존재

한국어에서는 **토큰화**를 통해 접사나 조사 등을 분리하는 것은 중요한 작업

---

### 한국어는 띄어쓰기가 제대로 지켜지지 않는다.

한국어는 띄어쓰기를 제대로 하지 않아도 의미가 전달되며, 띄어쓰기 규칙 또한 상대적으로 까다로운 언어

토큰이 제대로 분리 되지 않은 채 훈련 데이터로 사용된다면 언어 모델은 제대로 동작하지 않는다.

---

## 펄플렉서티 (Perplexity)

두 개의 모델 A, B가 있을 때 성능은 어떻게 비교할 수 있을까?

**외부 평가(extrinsic evaluation)**

두 모델의 성능을 비교하고자, 일일이 모델들에 대해서 실제 작업을 시켜보고 정확도를 비교하는 작업은 공수가 너무 많이 드는 작업

**내부 평가(Intrinsic evaluation)**

조금은 부정확할 수는 있어도 테스트 데이터에 대해서 빠르게 식으로 계산되는 더 간단한 평가 방법

모델 내에서 자신의 성능을 수치화하여 결과를 내놓는 평가

**perplexity** 

### 1. 언어 모델의 평가 방법 (Evaluation metric) : Perplexity 줄여서 PPL

언어 모델을 평가하기 위한 내부 평가 지표

단어의 수로 정규화(normalization) 된 테스트 데이터에 대한 확률의 역수

$$PPL(W)=P(w_{1}, w_{2}, w_{3}, ... , w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}, w_{2}, w_{3}, ... , w_{N})}}$$

**PPL을 최소화한다는 것은 문장의 확률을 최대화**

문장의 확률에 체인룰(chain rule)을 적용하면

$$PPL(W)=\sqrt[N]{\frac{1}{P(w_{1}, w_{2}, w_{3}, ... , w_{N})}}=\sqrt[N]{\frac{1}{\prod_{i=1}^{N}P(w_{i}| w_{1}, w_{2}, ... , w_{i-1})}}$$

### 2. 분기 계수(Branching factor)

PPL은 선택할 수 있는 가능한 경우의 수를 의미하는 분기계수(branching factor)이다.

PPL은 이 언어 모델이 특정 시점에서 평균적으로 **몇 개의 선택지**를 가지고 고민하고 있는지를 의미

ex) 언어 모델에 어떤 테스트 데이터을 주고 측정했더니 **PPL이 10**

 → 해당 언어 모델은 테스트 데이터에 대해서 다음 단어를 예측하는 모든 시점(time-step)마다 **평균적으로 10개의 단어**를 가지고 어떤 것이 정답인지 고민하고 있다고 볼 수 있다.

$$PPL(W)=P(w_{1}, w_{2}, w_{3}, ... , w_{N})^{-\frac{1}{N}}=(\frac{1}{10}^{N})^{-\frac{1}{N}}=\frac{1}{10}^{-1}=10$$

주의할 점!

PPL의 값이 낮다는 것은 테스트 데이터 상에서 높은 정확도를 보인다는 것이지, 

사람이 직접 느끼기에 좋은 언어 모델이라는 것을 반드시 의미하진 않는다는 점

정량적으로 양이 많고, 또한 도메인에 알맞은 동일한 테스트 데이터를 사용해야 신뢰도가 높다

### 3. 기존 언어 모델 vs 인공 신경망을 이용한 언어 모델

<img src = "/image/Untitled 7.png" width = "600px">

---

# 카운트 기반의 단어 표현 (Count based word Representation)

머신 러닝 등의 알고리즘이 적용된 본격적인 자연어 처리를 위해서는 

**문자를 숫자로 수치화 할 필요가 있다.**

## 다양한 단어의 표현 방법

### 1. 단어의 표현 방법

**국소 표현(Local Representation)** 방법

  해당 단어 그 자체만 보고, 특정값을 맵핑하여 단어를 표현하는 방법

**분산 표현(Distributed Representation)** 방법

  그 단어를 표현하고자 주변을 참고하여 단어를 표현하는 방법

ex) puppy, cute, lovely  ← 이 세 단어를 

국소 표현 예) 1번 2번 3번 등과 같은 숫자를 mapping 하여 부여

분산 표현 예) puppy 단어 근처에 cute, lovely 자주 등장 

                   → puppy 는 cute, lovely 한 느낌으로 정의 

단어의 의미, 뉘앙스를 표현  : 국소 표현 O  /  분산 표현 X

국소 표현 방법(Local Representation)을 이산 표현(Discrete Representation)라고,

분산 표현(Distributed Representation)을 연속 표현(Continuous Representation)라고도한다

---

### 2. 단어 표현의 카테고리화

<img src = "/image/Untitled 8.png" width = "600px">

**Bag of Words** :    국소 표현에(Local Representation)에 속하며, 단어의 빈도수를 카운트(Count)하여 단어를 수치화하는 단어 표현 방법

**DTM**(또는 **TDM**):   BoW 의 확장

**TF-IDF** :    빈도수 기반 단어 표현에 단어의 중요도에 따른 가중치를 줌 

**LSA**:    단어의 뉘앙스를 반영하는 연속 표현(Continuous Representation)의 일종

**Word2Vec**(워드투벡터) :    연속 표현(Continuous Representation)에 속하면서, 예측(prediction)을 기반으로 단어의 뉘앙스를 표현

**FastText**(패스트텍스트) :      Word2Vec 의 확장 

**GloVe**(글로브):     예측과 카운트라는 두 가지 방법이 모두 사용

---

## Bag of Words(BoW)

단어의 **등장 순서를 고려하지 않는** 빈도수 기반의 단어 표현 방법

단어들의 **출현 빈도(frequency)**에만 집중

BoW를 만드는 과정

(1) 우선, 각 단어에 고유한 정수 인덱스를 부여

(2) 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터를 만든다.

ex) 

문서1:  정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.

입력된 문서에 대해서 단어 집합(vocaburary)을 만들어 인덱스를 할당하고, BoW를 만드는 코드

```python
from konlpy.tag import Okt
import re  
okt=Okt()  

token=re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")  
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.  
token=okt.morphs(token)  
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.  

word2index={}  
bow=[]  
for voca in token:  
         if voca not in word2index.keys():  
             word2index[voca]=len(word2index)  
# token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.   
             bow.insert(len(word2index)-1,1)
# BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.  
         else:
            index=word2index.get(voca)
# 재등장하는 단어의 인덱스를 받아옵니다.
            bow[index]=bow[index]+1
# 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)  

print(word2index)  
('정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9)  

bow  
[1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
```

---

### 2. Bag of Words 의 다른 예제들

BoW에 있어서 **중요**한 것은 **단어의 등장 빈도**

**단어의 순서**. 즉, 인덱스의 순서는 전혀 **상관없다!**

ex) 

  문서 2: 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.

```python
('소비자': 0, '는': 1, '주로': 2, '소비': 3, '하는': 4, '상품': 5, '을': 6, '기준': 7, '으로': 8, '물가상승률': 9, '느낀다': 10)  
[1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
```

문서 1과 문서 2를 합쳐 

  문서3: 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다. 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.

```python
('정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9, '는': 10, '주로': 11, '소비': 12, '상품': 13, '을': 14, '기준': 15, '으로': 16, '느낀다': 17)  
[1, 2, 1, 2, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
```

BoW는 종종 여러 문서의 단어 집합을 합친 뒤에, 해당 단어 집합에 대한 각 문서의 BoW를 구하기도 한다. 

문서3에 대한 단어 집합을 기준으로 문서1, 문서2의 BoW를 만든다고 한다면

```python
문서3 단어 집합에 대한 문서1 BoW : [1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  
문서3 단어 집합에 대한 문서2 BoW : [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 1]
```

BoW는 각 단어가 등장한 횟수를 수치화 하는 텍스트 표현 방법이기 때문에, 
주로 어떤 단어가 얼마나 등장 했는지를 기준으로 
문서가 어떤 성격의 문서 인지를 판단하는 작업에 쓰인다.
즉, 분류 문제나 여러 문서 간의 유사도를 구하는 문제에 주로 쓰인다. 

ex)
'미분', '방정식', '부등식'과 같은 단어가 자주 등장한다면 
수학 관련 문서로 분류할 수 있다.
'달리기', '체력', '근력'과 같은 단어가 자주 등장하면 
해당 문서를 체육 관련 문서로 분류할 수 있다. 

---

### 3. CounVectorizer 클래스로 BoW 만들기

사이킷 런에서 단어의 빈도를 Count하여 Vector로 만드는 **CountVectorizer** 클래스 지원

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.

[[1 1 2 1 2 1]]
{'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
```

'I' 가 사라진 이유?

CountVectorizer가 기본적으로 길이가 2이상인 문자에 대해서만 토큰으로 인식하기 때문

주의! 
단지 띄어쓰기만을 기준으로 단어를 자르는 낮은 수준의 토큰화를 진행하고 BoW를 만든다

한국어에 CountVectorizer를 적용하면, 조사 등의 이유로 제대로 BoW가 만들어지지 않음을 의미

---

### 4. 불용어를 제거한 BoW 만들기

BoW를 사용한다는 것은 그 문서에서 각 단어가 얼마나 자주 등장했는지를 보겠다는 것

BoW를 만들 때 불용어를 제거하는 일 → 자연어 처리의 정확도를 높이기 위해서 선택할 수 있는 전처리 기법

**1) 사용자가 직접 정의한 불용어 사용**

```python
from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)
[[1 1 1 1 1]]
{'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
```

**2) CountVectorizer 에서 제공하는 자체 불용어 사용**

```python
from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)
[[1 1 1]]
{'family': 0, 'important': 1, 'thing': 2}
```

**3) NLTK에서 지원하는 불용어 사용**

```python
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

text=["Family is not an important thing. It's everything."]
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)
print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)
[[1 1 1 1]]
{'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
```

---

## 문서 단어 행렬 (Document - Term Matrix, DTM)

각 문서에 대한 BoW 표현 방법을 그대로 갖고 와서,

**서로 다른 문서들의 BoW들을 결합**한 표현 방법

### 1. 문서 단어 행렬 (Document - Term Matrix, DTM) 의 표기법

문서 단어 행렬(Document-Term Matrix, DTM)이란?

다수의 문서에서 등장하는 각 단어들의 빈도를 행렬로 표현한 것 

ex) 

문서1 : 먹고 싶은 사과

문서2 : 먹고 싶은 바나나

문서3 : 길고 노란 바나나 바나나

문서4 : 저는 과일이 좋아요

<img src = "/image/Untitled 9.png" width = "600px">

---

### 2. 문서 단어 행렬의 한계

**1) 희소 표현 (Sparse representation)**

**원-핫 벡터의 단점**과 마찬가지로 

각 문서 벡터의 차원은 원-핫 벡터와 마찬가지로 전체 단어 집합의 크기를 가진다.

또한 많은 문서 벡터가 대부분의 값이 0을 가질 수도 있다.

**희소 벡터(sparse vector)**  :  대부분의 값이 0인 표현 

→ 전처리를 통해 단어 집합의 크기를 줄이는 일은 BoW 표현을 사용하는 모델에서 중요!

ex) 구두점, 빈도수 낮은 단어, 불용어를 제거 / 어간이나 표제어 추출로 단어를 정규화

**2) 단순 빈도 수 기반 접근**

불용어인 the는 어떤 문서이든 자주 등장할 수 밖에 없다.

동일하게 the가 빈도수가 높다고 해서 이 문서들이 유사한 문서라고 판단해서는 안 된다!

각 문서에는 중요한 단어와 불필요한 단어들이 혼재 되어있다. 

DTM에 불용어와 중요한 단어에 대해 **가중치를 줄 수 있는 방법**은 없을까? 

이를 위해 사용하는 것이 TF-IDF입니다.

---

## TF-IDF (Term Frequency-Inverse Document Freqency)

TF-IDF를 사용하면, 

기존의 DTM을 사용하는 것보다 보다 더 많은 정보를 고려하여 문서들을 비교할 수 있다.

주의! TF-IDF가 DTM보다 항상 성능이 뛰어나진 않다!

### 1. TF - IDF (단어 빈도- 역 문서 빈도, Term Freqency - Inverse Document Frequency)

TF-IDF  :    **TF와 IDF를 곱한 값**

우선 DTM을 만든 후, TF-IDF 가중치를 부여

주로 문서의 유사도를 구하는 작업, 검색 시스템에서 검색 결과의 중요도를 정하는 작업, 

문서 내에서 특정 단어의 중요도를 구하는 작업 등에 쓰일 수 있다.

**1) tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수**

**2) df(t) : 특정 단어 t가 등장한 문서의 수**

**3) idf(d, t) : df(t)에 반비례하는 수**

$$idf(d, t) = log(\frac{n}{1+df(t)})$$

**log**를 사용하지 않았을 때

→ 총 문서의 수 n이 커질 수록, IDF의 값은 기하급수적으로 커진다.

→ 희귀 단어들에 엄청난 가중치가 부여될 수 있다.

**분모에 1을 더해주는 이유**: 특정 단어가 전체 문서에서 등장하지 않을 경우에 분모가 0이 되는 상황을 방지하기 위함

**모든 문서**에서 자주 등장하는 단어는 **중요도가 낮**다고 판단
**특정 문서**에서만 자주 등장하는 단어는 **중요도가 높다**고 판단

TF-IDF 값이 낮으면 중요도가 낮은 것

불용어의 TF-IDF의 값은 다른 단어의 TF-IDF에 비해서 낮다.

### 2. 파이썬으로 TF-IDF 직접 구현하기

```python
import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
] 
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

N = len(docs) # 총 문서의 수

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df + 1))

def tfidf(t, d):
    return tf(t,d)* idf(t)
```

```python
result = []
for i in range(N): # 각 문서에 대해서 아래 명령을 수행
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]        
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = vocab)
tf_
```

<img src = "/image/Untitled 10.png" width = "600px">

```python
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
idf_
```

<img src = "/image/Untitled 11.png" width = "600px">

```python
result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]

        result[-1].append(tfidf(t,d))

tfidf_ = pd.DataFrame(result, columns = vocab)
tfidf_
```

<img src = "/image/Untitled 12.png" width = "600px">

여전히 문제점이 존재

log항의 분자와 분모의 값이 같아질 수 있다. 

→ log의 진수값이 1이 되면서 idf(d,t)의 값이 0이 됨을 의미

그래서 실제 구현체는 log항에 1을 더해 IDF가 최소 1이상의 값을 가지도록 한다.

$$idf(d, t) = log(n/(df(t)+1)) + 1$$

---

### 3. 사이킷런을 이용한 DTM과 TF-IDF 실습

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.

[[0 1 0 1 0 1 0 1 1]
 [0 0 1 0 0 0 0 1 0]
 [1 0 0 0 1 0 1 0 0]]
{'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
```

사이킷런은 TF-IDF를 자동 계산해주는 **TfidfVectorizer** 제공한다.

사이킷런의 TF-IDF는 위에서 배웠던 보편적인 TF-IDF 식에서 좀 더 조정된 다른 식을 사용

(로그항의 분자에 1을 더해주며, 로그항에 1을 더해주고, TF-IDF에 L2 정규화라는 방법으로 값을 조정하는 등의 차이)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)

[[0.         0.46735098 0.         0.46735098 0.         0.46735098 0.         0.35543247 0.46735098]
 [0.         0.         0.79596054 0.         0.         0.         0.         0.60534851 0.        ]
 [0.57735027 0.         0.         0.         0.57735027 0.         0.57735027 0.         0.        ]]
{'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
```

---

# 문서 유사도 (Document Similarity)

자연어 처리의 주요 주제 중 하나

각 문서의 단어들을 어떤 방법으로 수치화하여 표현했는지**(DTM, Word2Vec 등)**, 

문서 간의 단어들의 차이를 어떤 방법**(유클리드 거리, 코사인 유사도 등)**으로 계산했는지에 달려있다.

## 코사인 유사도 (Cosine Similarity)

### 1. 코사인 유사도

두 벡터 간의 **코사인 각도**를 이용

<img src = "/image/Untitled 13.png" width = "600px">

**-1 이상 1 이하의 값**을 가지며 값이 1에 가까울수록 유사도가 높다고 판단

$$similarity=cos(Θ)=\frac{A⋅B}{||A||\ ||B||}=\frac{\sum_{i=1}^{n}{A_{i}×B_{i}}}{\sqrt{\sum_{i=1}^{n}(A_{i})^2}×\sqrt{\sum_{i=1}^{n}(B_{i})^2}}$$

ex) 

문서1 : 저는 사과 좋아요

문서2 : 저는 바나나 좋아요

문서3 : 저는 바나나 좋아요 저는 바나나 좋아요

세 문서에 대한 DTM 

<img src = "/image/Untitled 14.png" width = "600px">

```python
from numpy import dot
from numpy.linalg import norm
import numpy as np
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

doc1=np.array([0,1,1,1])
doc2=np.array([1,0,1,1])
doc3=np.array([2,0,2,2])

print(cos_sim(doc1, doc2)) #문서1과 문서2의 코사인 유사도
print(cos_sim(doc1, doc3)) #문서1과 문서3의 코사인 유사도
print(cos_sim(doc2, doc3)) #문서2과 문서3의 코사인 유사도

0.67
0.67
1.00
```

코사인 유사도는 문서의 길이가 다른 상황에서 비교적 공정한 비교를 할 수 있도록 도와준다.

벡터의 크기가 아니라 벡터의 방향(패턴)에 초점을 두기 때문!

---

### 2. 유사도를 이용한 추천 시스템 구현하기

[https://www.kaggle.com/rounakbanik/the-movies-dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)

movies_metadata.csv     총 24개의 열을 가진 45,466개의 샘플로 구성

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
data = pd.read_csv('현재 movies_metadata.csv의 파일 경로', low_memory=False)
data.head(2)
```

<img src = "/image/Untitled 15.png" width = "600px">

```python
data = data.head(20000)
```

tf-idf 할 때 데이터에 Null 값 들어있으면 에러 발생 → Null 대신 넣고자 하는 값으로 대체

```python
data['overview'].isnull().sum()
135

# overview에서 Null 값을 가진 경우에는 Null 값을 제거
data['overview'] = data['overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
# overview에 대해서 tf-idf 수행
tfidf_matrix = tfidf.fit_transform(data['overview'])
print(tfidf_matrix.shape)

(20000, 47487)
# 20,000개의 영화를 표현하기위해 총 47,487개의 단어가 사용되었다.
```

코사인 유사도를 사용하여 바로 문서의 유사도를 구할 수 있다.

```python
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# 코사인 유사도를 구한다.

indices = pd.Series(data['title'].drop_duplicates().index, index=data['title'].drop_duplicates())
print(indices.head())

title
Toy Story                      0
Jumanji                        1
Grumpier Old Men               2
Waiting to Exhale              3
Father of the Bride Part II    4
dtype: int64

idx = indices['Father of the Bride Part II']
print(idx)
4

def get_recommendations(title, cosine_sim=cosine_sim):
    
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return data['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises')
12481                            The Dark Knight
150                               Batman Forever
1328                              Batman Returns
15511                 Batman: Under the Red Hood
585                                       Batman
9230          Batman Beyond: Return of the Joker
18035                           Batman: Year One
19792    Batman: The Dark Knight Returns, Part 1
3095                Batman: Mask of the Phantasm
10122                              Batman Begins
Name: title, dtype: object
```

---

## 여러가지 유사도 기법

문서의 유사도를 구하기 위한 방법으로는 코사인 유사도 외에도 여러가지 방법들이 있다.

### 1. 유클리드 거리 (Euclidean distance)

자카드 유사도나 코사인 유사도만큼, 유용한 방법은 아니다.

다차원 공간에서 두개의 점 p와 q가 각각 p=(p1,p2,p3,...,pn)과 q=(q1,q2,q3,...,qn)의 좌표를 가질 때 두 점 사이의 거리를 계산하는 유클리드 거리 공식

$$\sqrt{(q_{1}-p_{1})^{2}+(q_{2}-p_{2})^{2}+\ ...\ +(q_{n}-p_{n})^{2}}=\sqrt{\sum_{i=1}^{n}(q_{i}-p_{i})^{2}}$$

ex)

<img src = "/image/Untitled 16.png" width = "600px">

<img src = "/image/Untitled 17.png" width = "600px">

이때 다음과 같은 문서Q에 대해서 문서1, 문서2, 문서3 중 가장 유사한 문서를 찾는다면

```python
import numpy as np
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

doc1 = np.array((2,3,0,1))
doc2 = np.array((1,2,3,1))
doc3 = np.array((2,1,2,2))
docQ = np.array((1,1,0,1))

print(dist(doc1,docQ))
print(dist(doc2,docQ))
print(dist(doc3,docQ))
2.23606797749979
3.1622776601683795
2.449489742783178
```

유클리드 거리의 값이 가장 작다는 것은, 

→ 문서 간의 거리가 가장 가깝다는 것.

 즉, 문서1이 문서Q와 가장 유사하다고 볼 수 있다.

---

### 2. 자카드 유사도 (Jaccard similarity)

자카드 유사도(jaccard similarity)의 아이디어

→    합집합에서 교집합의 비율을 구한다면 두 집합 A와 B의 유사도를 구할 수 있다!

$$J(A,B)=\frac{|A∩B|}{|A∪B|}=\frac{|A∩B|}{|A|+|B|-|A∩B|}$$

0과 1사이의 값을 갖는다!

만약 두 집합이 동일하면 1의 값을 가지고, 두 집합의 공통 원소가 없다면 0의 값을 갖는다.

$$J(doc_{1},doc_{2})=\frac{doc_{1}∩doc_{2}}{doc_{1}∪doc_{2}}$$

```python
# 다음과 같은 두 개의 문서가 있습니다.
# 두 문서 모두에서 등장한 단어는 apple과 banana 2개.
doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

# 토큰화를 수행합니다.
tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

# 토큰화 결과 출력
print(tokenized_doc1)
print(tokenized_doc2)

['apple', 'banana', 'everyone', 'like', 'likey', 'watch', 'card', 'holder']
['apple', 'banana', 'coupon', 'passport', 'love', 'you']

union = set(tokenized_doc1).union(set(tokenized_doc2))
print(union)
{'card', 'holder', 'passport', 'banana', 'apple', 'love', 'you', 'likey', 'coupon', 'like', 'watch', 'everyone'}

intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
print(intersection)
{'banana', 'apple'}

print(len(intersection)/len(union)) # 2를 12로 나눔.
0.16666666666666666

# 자카드 유사도이자, 
# 두 문서의 총 단어 집합에서 두 문서에서 공통적으로 등장한 단어의 비율이다.
```

---

# 토픽 모델링 (Topic Modeling)

기계 학습 및 자연어 처리 분야에서 토픽이라는 문서 집합의 추상적인 주제를 발견하기 위한 통계적 모델 중 하나로, 텍스트 본문의 숨겨진 의미 구조를 발견하기 위해 사용되는 텍스트 마이닝 기법

## 잠재 의미 분석 (Latent Semantic Analysis, LSA)

LSA는 토픽 모델링 분야에 아이디어를 제공한 알고리즘이라고 볼 수 있다!

뒤에 나오는 **LDA**는 **LSA의 단점**을 개선해 토픽 모델링에 보다 적합한 알고리즘!!

BoW에 기반한 DTM이나 TF-IDF는 **단어의 빈도 수**를 이용!
→ 단어의 의미를 고려하지 못한다는 단점이 있다!

DTM의 잠재된(Latent) 의미를 이끌어내는 방법 → LSA 또는 LSI (잠재 의미 분석)

LSA 를 이해하기 위해 **특이값 분해**를 이해해야 한다!

### 1. 특이값 분해 (Singular Value Decomposition, SVD)

실수 벡터 공간에 한정

A가 m × n 행렬일 때, 다음과 같이 3개의 행렬의 곱으로 분해(decomposition)

$$A=UΣV^\text{T}$$

$U: m × m\ \text{직교행렬}\ (AA^\text{T}=U(ΣΣ^\text{T})U^\text{T})$

$V: n × n\ \text{직교행렬}\ (A^\text{T}A=V(Σ^\text{T}Σ)V^\text{T})$

$Σ: m × n\ \text{직사각 대각행렬}$

SVD로 나온 **대각 행렬의 대각 원소의 값**이  행렬 A의 **특이값(singular value)**

SVD를 통해 나온 대각 행렬 Σ의 추가적인 성질!

    → 특이값들이 **내림차순으로 정렬**되어 있다!

---

### 2. 절단된 SVD (Truncated SVD)

LSA의 경우 **풀 SVD**의 3개의 행렬에서 **일부 벡터들을 삭제** 시킨 **절단된 SVD(truncated SVD)**를 사용

<img src = "/image/Untitled 18.png" width = "600px">

절단된 SVD는 대각 행렬 Σ의 대각 원소의 값 중에서 **상위 값 t개**만 남게 된다.

U행렬과 V행렬의 **t열**까지만 남긴다. 

💡여기서 t는 우리가 찾고자 하는 **토픽의 수**를 반영한 **하이퍼파라미터!**

**t를 크게 잡으면** 기존의 행렬 A로부터 다양한 의미를 가져갈 수 있다!

그러나 ! **t를 작게 잡아야만** 노이즈를 제거할 수 있다! 

이렇게 일부 벡터들을 삭제하는 것 → '**데이터의 차원을 줄인다!**'

풀 SVD 보다 직관적으로 계산 비용이 낮아진다!
상대적으로 중요하지 않은 정보를 삭제하는 효과가 있다! 
    영상 처리 분야: 노이즈를 제거한다는 의미
    자연어 처리 분야: 설명력 낮은 정보를 삭제, 설명력 높은 정보를 남긴다는 의미

기존의 행렬에서는 드러나지 않았던 심층적인 의미를 확인할 수 있게 된다!

---

### 3. 잠재 의미 분석 (Latent Semantic Analysis , LSA)

<img src = "/image/Untitled 19.png" width = "600px">

위의 DTM을 numpy로 구현

```python
import numpy as np
A=np.array([[0,0,0,1,0,1,1,0,0],[0,0,0,1,1,0,1,0,0],[0,1,1,0,2,0,0,0,0],[1,0,0,0,0,0,0,1,1]])
np.shape(A)
(4, 9) # 4 x 9 크기의 DTM
```

**풀 SVD(full SVD)**

4 × 4의 크기를 가지는 직교 행렬 U 생성

```python
U, s, VT = np.linalg.svd(A, full_matrices = True)
print(U.round(2))
np.shape(U)

[[-0.24  0.75  0.   -0.62]
 [-0.51  0.44 -0.    0.74]
 [-0.83 -0.49 -0.   -0.27]
 [-0.   -0.    1.    0.  ]]
(4, 4)
```

대각 행렬 S 생성

Numpy의 linalg.svd()는 특이값 분해의 결과로 대각 행렬이 아니라 특이값의 리스트를 반환

```python
print(s.round(2))
np.shape(s)
[2.69 2.05 1.73 0.77]
(4,)
```

이를 다시 대각 행렬의 형태로

```python
S = np.zeros((4, 9)) # 대각 행렬의 크기인 4 x 9의 임의의 행렬 생성
S[:4, :4] = np.diag(s) # 특이값을 대각행렬에 삽입
print(S.round(2))
np.shape(S)
[[2.69 0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   2.05 0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   1.73 0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.77 0.   0.   0.   0.   0.  ]]   # 특이값들 내림차순 확인 가능!
(4, 9)
```

9 × 9의 크기를 가지는 직교 행렬 VT(V의 전치 행렬) 생성

```python
print(VT.round(2))
np.shape(VT)
[[-0.   -0.31 -0.31 -0.28 -0.8  -0.09 -0.28 -0.   -0.  ]
 [ 0.   -0.24 -0.24  0.58 -0.26  0.37  0.58 -0.   -0.  ]
 [ 0.58 -0.    0.    0.   -0.    0.   -0.    0.58  0.58]
 [ 0.   -0.35 -0.35  0.16  0.25 -0.8   0.16 -0.   -0.  ]
 [-0.   -0.78 -0.01 -0.2   0.4   0.4  -0.2   0.    0.  ]
 [-0.29  0.31 -0.78 -0.24  0.23  0.23  0.01  0.14  0.14]
 [-0.29 -0.1   0.26 -0.59 -0.08 -0.08  0.66  0.14  0.14]
 [-0.5  -0.06  0.15  0.24 -0.05 -0.05 -0.19  0.75 -0.25]
 [-0.5  -0.06  0.15  0.24 -0.05 -0.05 -0.19 -0.25  0.75]]
(9, 9)
```

U × S × VT를 하면 기존의 행렬 A와 동일한지 확인

```python
np.allclose(A, np.dot(np.dot(U,S), VT).round(2))
True
```

이제 절단된 SVD(Truncated SVD)를 수행

t = 2  설정

대각 행렬 S 내의 특이값 중에서 상위 2개만 남기고 제거

```python
S=S[:2,:2]
print(S.round(2))
[[2.69 0.  ]
 [0.   2.05]]
```

직교 행렬 U 도 2개의 열만 남기고 제거

```python
U=U[:,:2]
print(U.round(2))
[[-0.24  0.75]
 [-0.51  0.44]
 [-0.83 -0.49]
 [-0.   -0.  ]]
```

VT에 대해서 2개의 행만 남기고 제거

(이는 V관점에서는 2개의 열만 남기고 제거한 것)

```python
VT=VT[:2,:]
print(VT.round(2))
[[-0.   -0.31 -0.31 -0.28 -0.8  -0.09 -0.28 -0.   -0.  ]
 [ 0.   -0.24 -0.24  0.58 -0.26  0.37  0.58 -0.   -0.  ]]
```

축소된 행렬 U, S, VT 으로 다시 U × S × VT  하면 기존의 A와 다른 결과가 나오게 된다!

```python
A_prime=np.dot(np.dot(U,S), VT)
print(A)
print(A_prime.round(2))
[[0 0 0 1 0 1 1 0 0]
 [0 0 0 1 1 0 1 0 0]
 [0 1 1 0 2 0 0 0 0]
 [1 0 0 0 0 0 0 1 1]]
[[ 0.   -0.17 -0.17  1.08  0.12  0.62  1.08 -0.   -0.  ]
 [ 0.    0.2   0.2   0.91  0.86  0.45  0.91  0.    0.  ]
 [ 0.    0.93  0.93  0.03  2.05 -0.17  0.03  0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.  ]]

# 대체적으로 기존에 0인 값들은 0에 가까운 값이 나오고, 
# 1인 값들은 1에 가까운 값이 나오는 것을 볼 수 있다!
# 값이 제대로 복구되지 않은 구간도 존재!
```

축소된 U는 4 × 2의 크기 
→ 문서의 개수 × 토픽의 수 t의 크기
→ 4개의 문서 각각을 2개의 값으로 표현
→ 즉, U의 각 행은 잠재 의미를 표현하기 위한 수치화 된 각각의 **문서 벡터**

축소된 VT는 2 × 9의 크기
→ 토픽의 수 t × 단어의 개수의 크기
→ 즉, VT의 각 열은 잠재 의미를 표현하기 위해 수치화 된 각각의 **단어 벡터

이 문서 벡터들과 단어 벡터들을 통해 
다른 문서의 유사도, 다른 단어의 유사도, 단어(쿼리)로부터 문서의 유사도를 구하는 것들이 가능**

---

### 4. 실습을 통한 이해

사이킷런에서 Twenty Newsgroups이라는 20개의 다른 주제를 가진 뉴스 그룹 데이터를 제공

LSA를 사용해 문서의 수를 원하는 토픽의 수로 압축해,

각 토픽당 가장 중요한 단어 5개를 출력하는 실습으로 토픽 모델링을 수행

**1) 뉴스그룹 데이터에 대한 이해**

```python
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
len(documents)
11314 # 훈련에 사용할 뉴스그룹 데이터는 총 11,314개

documents[1]
"\n\n\n\n\n\n\nYeah, do you expect people to read the FAQ, etc. and actually accept hard\natheism?  No, you need a little leap of faith, Jimmy.  Your logic runs out\nof steam!\n\n\n\n\n\n\n\nJim,\n\nSorry I can't pity you, Jim.  And I'm sorry that you have these feelings of\ndenial about the faith you need to get by.  Oh well, just pretend that it will\nall end happily ever after anyway.  Maybe if you start a new newsgroup,\nalt.atheist.hard, you won't be bummin' so much?\n\n\n\n\n\n\nBye-Bye, Big Jim.  Don't forget your Flintstone's Chewables!  :) \n--\nBake Timmons, III"
```

target_name에는 이 데이터가 어떤 20개의 카테고리를 갖고 있는 지 저장되어 있다.

```python
print(dataset.target_names)
['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
```

**2) 텍스트 전처리**

아이디어

- 알파벳 제외한 구두점, 숫자, 특수 문자 제거
- 길이가 짧은 단어 제거
- 모든 알파벳을 소문자로 바꿔 단어의 개수 줄이기

```python
news_df = pd.DataFrame({'document':documents})
# 특수 문자 제거
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# 전체 단어에 대한 소문자 변환
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

news_df['clean_doc'][1]
'yeah expect people read actually accept hard atheism need little leap faith jimmy your logic runs steam sorry pity sorry that have these feelings denial about faith need well just pretend that will happily ever after anyway maybe start newsgroup atheist hard bummin much forget your flintstone chewables bake timmons'
```

토큰화! 그 다음 불용어 제거!

```python
from nltk.corpus import stopwords
stop_words = stopwords.words('english') # NLTK로부터 불용어를 받기
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
# 불용어를 제거

print(tokenized_doc[1])
['yeah', 'expect', 'people', 'read', 'actually', 'accept', 'hard', 'atheism', 'need', 'little', 'leap', 'faith', 'jimmy', 'logic', 'runs', 'steam', 'sorry', 'pity', 'sorry', 'feelings', 'denial', 'faith', 'need', 'well', 'pretend', 'happily', 'ever', 'anyway', 'maybe', 'start', 'newsgroup', 'atheist', 'hard', 'bummin', 'much', 'forget', 'flintstone', 'chewables', 'bake', 'timmons']
```

**3) TF-IDF 행렬 만들기**

TfidfVectorizer는 기본적으로 토큰화가 되어있지 않은 텍스트 데이터를 입력으로 사용

다시 토큰화 작업을 역으로 취소하는 작업을 수행 

역토큰화 (Detokenization)

```python
# 역토큰화 (토큰화 작업을 역으로 되돌림)
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc

news_df['clean_doc'][1]
'yeah expect people read actually accept hard atheism need little leap faith jimmy logic runs steam sorry pity sorry feelings denial faith need well pretend happily ever anyway maybe start newsgroup atheist hard bummin much forget flintstone chewables bake timmons'
```

TfidfVectorizer를 통해 단어 1,000개에 대한 TF-IDF 행렬 생성

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000, # 상위 1,000개의 단어를 보존 
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(news_df['clean_doc'])
X.shape # TF-IDF 행렬의 크기 확인
(11314, 1000)
```

**4) 토픽 모델링 (Topic Modeling)**

사이킷런의 절단된 SVD(Truncated SVD)를 사용 → 차원 축소 가능!

기존 데이터가 20개의 카테고리를 갖고 있었기 때문에, 20개의 토픽을 가졌다고 가정

```python
from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)
len(svd_model.components_)
20 

np.shape(svd_model.components_) # LSA에서 VT에 해당
(20, 1000)
```

각 20개의 행의 각 1,000개의 열 중 가장 값이 큰 5개의 값을 찾아서 단어로 출력

```python
terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어가 저장됨.

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(svd_model.components_,terms)

Topic 1: [('like', 0.2138), ('know', 0.20031), ('people', 0.19334), ('think', 0.17802), ('good', 0.15105)]
Topic 2: [('thanks', 0.32918), ('windows', 0.29093), ('card', 0.18016), ('drive', 0.1739), ('mail', 0.15131)]
Topic 3: [('game', 0.37159), ('team', 0.32533), ('year', 0.28205), ('games', 0.25416), ('season', 0.18464)]
Topic 4: [('drive', 0.52823), ('scsi', 0.20043), ('disk', 0.15518), ('hard', 0.15511), ('card', 0.14049)]
Topic 5: [('windows', 0.40544), ('file', 0.25619), ('window', 0.1806), ('files', 0.16196), ('program', 0.14009)]
Topic 6: [('government', 0.16085), ('chip', 0.16071), ('mail', 0.15626), ('space', 0.15047), ('information', 0.13582)]
Topic 7: [('like', 0.67121), ('bike', 0.14274), ('know', 0.11189), ('chip', 0.11043), ('sounds', 0.10389)]
Topic 8: [('card', 0.44948), ('sale', 0.21639), ('video', 0.21318), ('offer', 0.14896), ('monitor', 0.1487)]
Topic 9: [('know', 0.44869), ('card', 0.35699), ('chip', 0.17169), ('video', 0.15289), ('government', 0.15069)]
Topic 10: [('good', 0.41575), ('know', 0.23137), ('time', 0.18933), ('bike', 0.11317), ('jesus', 0.09421)]
Topic 11: [('think', 0.7832), ('chip', 0.10776), ('good', 0.10613), ('thanks', 0.08985), ('clipper', 0.07882)]
Topic 12: [('thanks', 0.37279), ('right', 0.21787), ('problem', 0.2172), ('good', 0.21405), ('bike', 0.2116)]
Topic 13: [('good', 0.36691), ('people', 0.33814), ('windows', 0.28286), ('know', 0.25238), ('file', 0.18193)]
Topic 14: [('space', 0.39894), ('think', 0.23279), ('know', 0.17956), ('nasa', 0.15218), ('problem', 0.12924)]
Topic 15: [('space', 0.3092), ('good', 0.30207), ('card', 0.21615), ('people', 0.20208), ('time', 0.15716)]
Topic 16: [('people', 0.46951), ('problem', 0.20879), ('window', 0.16), ('time', 0.13873), ('game', 0.13616)]
Topic 17: [('time', 0.3419), ('bike', 0.26896), ('right', 0.26208), ('windows', 0.19632), ('file', 0.19145)]
Topic 18: [('time', 0.60079), ('problem', 0.15209), ('file', 0.13856), ('think', 0.13025), ('israel', 0.10728)]
Topic 19: [('file', 0.4489), ('need', 0.25951), ('card', 0.1876), ('files', 0.17632), ('problem', 0.1491)]
Topic 20: [('problem', 0.32797), ('file', 0.26268), ('thanks', 0.23414), ('used', 0.19339), ('space', 0.13861)]
```

---

### 5.  LSA의 장단점 (Pros and Cons of LSA)

쉽고 빠르게 구현이 가능

단어의 잠재적인 의미를 이끌어낼 수 있다 
  → 문서 유사도 계산 등에서 좋은 성능 보여준다! 

SVD의 특성 상 이미 계산된 LSA에 새로운 데이터를 추가해 계산하려면 처음부터 다시 계산해야 한다. 
  →즉, 새로운 정보에 대해 업데이트가 어렵다!

**최근 LSA 대신 Word2Vec 등 단어의 의미를 벡터화 할 수 있는 또 다른 방법론인 인공 신경망 기반의 방법론이 각광받는 이유**

---

## 잠재 디리클레 할당 (Latent Dirichlet Allocation, LDA)

토픽 모델링의 대표적인 알고리즘!

문서들은 토픽들의 혼합으로 구성되어 있으며, 토픽들은 확률 분포에 기반하여 단어들을 생성한다고 가정

데이터가 주어지면, LDA는 문서가 생성되던 과정을 역추적

### 1. 잠재 디리클레 할당 (Latent Dirichlet Allocation, LDA) 개요

ex)

3개의 문서 집합을 입력하면 어떤 결과를 보여주는지 간소화 된 예

문서1 : 저는 사과랑 바나나를 먹어요

문서2 : 우리는 귀여운 강아지가 좋아요

문서3 : 저의 깜찍하고 귀여운 강아지가 바나나를 먹어요

문서 집합에서 **토픽이 몇 개가 존재할지 가정**하는 것은 사용자가 해야 할 일

LDA에 2개의 토픽을 찾으라고 요청

전처리 과정을 거친 DTM이 LDA의 입력이 되었다고 가정

세 문서로부터 2개의 토픽을 찾은 결과

LDA는 **각 문서의 토픽 분포**와 **각 토픽 내의 단어 분포**를 추정

**<각 문서의 토픽 분포>**

문서1 : 토픽 A 100%

문서2 : 토픽 B 100%

문서3 : 토픽 B 60%, 토픽 A 40%

**<각 토픽의 단어 분포>**

토픽A : **사과 20%, 바나나 40%, 먹어요 40%**, 귀여운 0%, 강아지 0%, 깜찍하고 0%, 좋아요 0%

토픽B : 사과 0%, 바나나 0%, 먹어요 0%, **귀여운 33%, 강아지 33%, 깜찍하고 16%, 좋아요 16%**

→ 사용자는 위 결과로부터 두 토픽이 각각 과일에 대한 토픽과 강아지에 대한 토픽이라고 판단할 수 있다.

---

### 2. LDA의 가정

DTM 또는 TF-IDF 행렬을 입력으로 한다! → LDA는 단어의 순서는 신경 쓰지 않는다!

**'나는 이 문서를 작성하기 위해서 이런 주제들을 넣을거고, 이런 주제들을 위해서는 이런 단어들을 넣을 거야.'**

**1) 문서에 사용할 단어의 개수 N을 정한다**

**2) 문서에 사용할 토픽의 혼합을 확률 분포에 기반하여 결정한다

    -** Ex) 위 예제와 같이 토픽이 2개라고 했을 때 강아지 토픽을 60%, 과일 토픽을 40%와 같이 선택할 수 있다

**3) 문서에 사용할 각 단어를 (아래와 같이) 정한다

3-1) 토픽 분포에서 토픽 T를 확률적으로 고른다**

    - Ex) 60% 확률로 강아지 토픽을 선택하고, 40% 확률로 과일 토픽을 선택할 수 있다.

**3-2) 선택한 토픽 T에서 단어의 출현 확률 분포에 기반해 문서에 사용할 단어를 고릅니다.**

    - Ex) 강아지 토픽을 선택했다면, 33% 확률로 강아지란 단어를 선택할 수 있다. 3)을 반복하면서 문서를 완성한다.

이러한 과정을 통해 문서가 작성되었다는 가정 하에 

LDA는 토픽을 뽑아내기 위해 위 과정을 역으로 추적하는 **역공학(reverse engineering)** 수행

---

### 3. LDA 수행하기

**1) 사용자는 토픽의 개수 k를 설정**

k를 입력받으면, k개의 토픽이 M개의 전체 문서에 걸쳐 분포되어 있다고 가정

**2) 모든 단어를 k개 중 하나의 토픽에 할당**

이 작업이 끝나면 각 문서는 토픽을 가지며, 토픽은 단어 분포를 가지는 상태

물론 랜덤으로 할당 → 결과는 전부 틀린 상태

**3) 이제 모든 문서의 모든 단어에 대해서 아래의 사항을 반복 진행 (iterative)**

**3-1) 어떤 문서의 각 단어 w는 자신은 잘못된 토픽에 할당되어 있지만, 다른 단어들은 전부 올바른 토픽에 할당되어 있는 상태라고 가정. 이에 따라 단어 w는 아래의 두 가지 기준에 따라서 토픽이 재할당된다**

- p(topic t | document d) : 문서 d의 단어들 중 토픽 t에 해당하는 단어들의 비율
- p(word w | topic t) : 각 토픽들 t에서 해당 단어 w의 분포

ex)

doc1의 세번째 단어 apple의 토픽을 결정하고자 한다.

![https://wikidocs.net/images/page/30708/lda1.PNG](https://wikidocs.net/images/page/30708/lda1.PNG)

**첫 번째 기준은 문서 doc1의 단어들이 어떤 토픽에 해당하는지**

![https://wikidocs.net/images/page/30708/lda3.PNG](https://wikidocs.net/images/page/30708/lda3.PNG)

토픽 A와 토픽 B에 50 대 50의 비율로 할당 → 어느 토픽에도 속할 가능성이 있다!

**두번째 기준은 단어 apple이 전체 문서에서 어떤 토픽에 할당되어 있는지**

![https://wikidocs.net/images/page/30708/lda2.PNG](https://wikidocs.net/images/page/30708/lda2.PNG)

단어 apple은 토픽 B에 할당될 가능성이 높다! 

---

### 4. 잠재 디리클레 할당과 잠재 의미 분석의 차이

**LSA : DTM 또는 TF-IDF를 차원 축소 해 축소 차원에서 근접 단어들을 토픽으로 묶는다.**

**LDA : 단어가 특정 토픽에 존재할 확률과 문서에 특정 토픽이 존재할 확률을 결합확률로 추정하여 토픽을 추출한다.**

 

---

### 5. 실습

**1) 뉴스 기사 제목 데이터에 대한 이해**

약 15년 동안 발행되었던 뉴스 기사 제목을 모아 놓은 영어 데이터

링크 : [https://www.kaggle.com/therohk/million-headlines](https://www.kaggle.com/therohk/million-headlines)

```python
import pandas as pd
import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/franciscadias/data/master/abcnews-date-text.csv", filename="abcnews-date-text.csv")
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)

print(len(data))
1082168

print(data.head(5))
   publish_date                                      headline_text
0      20030219  aba decides against community broadcasting lic...
1      20030219     act fire witnesses must be aware of defamation
2      20030219     a g calls for infrastructure protection summit
3      20030219           air nz staff in aust strike for pay rise
4      20030219      air nz strike to affect australian travellers

text = data[['headline_text']]
text.head(5)
                                       headline_text
0  aba decides against community broadcasting lic...
1     act fire witnesses must be aware of defamation
2     a g calls for infrastructure protection summit
3           air nz staff in aust strike for pay rise
4      air nz strike to affect australian travellers
```

**2) 텍스트 전처리**

불용어 제거, 표제어 추출, 길이가 짧은 단어 제거

먼저 **단어 토큰화**

```python
import nltk
text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis=1)
NLTK의 word_tokenize를 통해 단어 토큰화를 수행합니다.

print(text.head(5))
                                       headline_text
0  [aba, decides, against, community, broadcastin...
1  [act, fire, witnesses, must, be, aware, of, de...
2  [a, g, calls, for, infrastructure, protection,...
3  [air, nz, staff, in, aust, strike, for, pay, r...
4  [air, nz, strike, to, affect, australian, trav...
```

**불용어 제거**

```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop)])

print(text.head(5))
                                       headline_text
0   [aba, decides, community, broadcasting, licence]
1    [act, fire, witnesses, must, aware, defamation]
2     [g, calls, infrastructure, protection, summit]
3          [air, nz, staff, aust, strike, pay, rise]
4  [air, nz, strike, affect, australian, travellers]
```

**표제어 추출**

```python
from nltk.stem import WordNetLemmatizer
text['headline_text'] = text['headline_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
print(text.head(5))
                                       headline_text
0       [aba, decide, community, broadcast, licence]
1      [act, fire, witness, must, aware, defamation]
2      [g, call, infrastructure, protection, summit]
3          [air, nz, staff, aust, strike, pay, rise]
4  [air, nz, strike, affect, australian, travellers]
```

**길이가 3이하인 단어 제거**

```python
tokenized_doc = text['headline_text'].apply(lambda x: [word for word in x if len(word) > 3])
print(tokenized_doc[:5])
0       [decide, community, broadcast, licence]
1      [fire, witness, must, aware, defamation]
2    [call, infrastructure, protection, summit]
3                   [staff, aust, strike, rise]
4      [strike, affect, australian, travellers]
```

**3) TF-IDF 행렬 만들기**

```python
# 역토큰화 (토큰화 작업을 되돌림)
detokenized_doc = []
for i in range(len(text)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

text['headline_text'] = detokenized_doc # 다시 text['headline_text']에 재저장

text['headline_text'][:5]
0       decide community broadcast licence
1       fire witness must aware defamation
2    call infrastructure protection summit
3                   staff aust strike rise
4      strike affect australian travellers
Name: headline_text, dtype: object

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000) # 상위 1,000개의 단어를 보존 
X = vectorizer.fit_transform(text['headline_text'])
X.shape # TF-IDF 행렬의 크기 확인
(1082168, 1000)

# 1,082,168 × 1,000의 크기를 가진 TF-IDF 행렬
```

**4) 토픽 모델링**

```python
from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=777,max_iter=1)
lda_top=lda_model.fit_transform(X)
print(lda_model.components_)
print(lda_model.components_.shape) 
[[1.00001533e-01 1.00001269e-01 1.00004179e-01 ... 1.00006124e-01
  1.00003111e-01 1.00003064e-01]
 [1.00001199e-01 1.13513398e+03 3.50170830e+03 ... 1.00009349e-01
  1.00001896e-01 1.00002937e-01]
 [1.00001811e-01 1.00001151e-01 1.00003566e-01 ... 1.00002693e-01
  1.00002061e-01 7.53381835e+02]
 ...
 [1.00001065e-01 1.00001689e-01 1.00003278e-01 ... 1.00006721e-01
  1.00004902e-01 1.00004759e-01]
 [1.00002401e-01 1.00000732e-01 1.00002989e-01 ... 1.00003517e-01
  1.00001428e-01 1.00005266e-01]
 [1.00003427e-01 1.00002313e-01 1.00007340e-01 ... 1.00003732e-01
  1.00001207e-01 1.00005153e-01]]
(10, 1000)
terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어가 저장

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(lda_model.components_,terms)
Topic 1: [('government', 8725.19), ('sydney', 8393.29), ('queensland', 7720.12), ('change', 5874.27), ('home', 5674.38)]
Topic 2: [('australia', 13691.08), ('australian', 11088.95), ('melbourne', 7528.43), ('world', 6707.7), ('south', 6677.03)]
Topic 3: [('death', 5935.06), ('interview', 5924.98), ('kill', 5851.6), ('jail', 4632.85), ('life', 4275.27)]
Topic 4: [('house', 6113.49), ('2016', 5488.19), ('state', 4923.41), ('brisbane', 4857.21), ('tasmania', 4610.97)]
Topic 5: [('court', 7542.74), ('attack', 6959.64), ('open', 5663.0), ('face', 5193.63), ('warn', 5115.01)]
Topic 6: [('market', 5545.86), ('rural', 5502.89), ('plan', 4828.71), ('indigenous', 4223.4), ('power', 3968.26)]
Topic 7: [('charge', 8428.8), ('election', 7561.63), ('adelaide', 6758.36), ('make', 5658.99), ('test', 5062.69)]
Topic 8: [('police', 12092.44), ('crash', 5281.14), ('drug', 4290.87), ('beat', 3257.58), ('rise', 2934.92)]
Topic 9: [('fund', 4693.03), ('labor', 4047.69), ('national', 4038.68), ('council', 4006.62), ('claim', 3604.75)]
Topic 10: [('trump', 11966.41), ('perth', 6456.53), ('report', 5611.33), ('school', 5465.06), ('woman', 5456.76)]
```

---

### 6. 실습 (gensim 사용)

Twenty Newsgroups이라고 불리는 20개의 다른 주제를 가진 뉴스 데이터를 다시 사용

동일한 전처리 과정을 거친 후에 tokenized_doc 으로 저장한 상태

**1) 정수 인코딩과 단어 집합 만들기**

```python
tokenized_doc[:5]
0    [well, sure, about, story, seem, biased, what,...
1    [yeah, expect, people, read, actually, accept,...
2    [although, realize, that, principle, your, str...
3    [notwithstanding, legitimate, fuss, about, thi...
4    [well, will, have, change, scoring, playoff, p...
Name: clean_doc, dtype: object
```

gensim의 corpora.Dictionary() 사용

→ 각 단어를 (word_id, word_frequency)의 형태로 손쉽게 바꾼다!

→ word_id 단어가 정수 인코딩된 값

→ word_frequency 해당 뉴스에서 해당 단어의 빈도수

```python
from gensim import corpora
dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus[1]) # 수행된 결과에서 두번째 뉴스 출력. 첫번째 문서의 인덱스는 0

[(52, 1), (55, 1), (56, 1), (57, 1), (58, 1), (59, 1), (60, 1), (61, 1), (62, 1), (63, 1), (64, 1), (65, 1), (66, 2), (67, 1), (68, 1), (69, 1), (70, 1), (71, 2), (72, 1), (73, 1), (74, 1), (75, 1), (76, 1), (77, 1), (78, 2), (79, 1), (80, 1), (81, 1), (82, 1), (83, 1), (84, 1), (85, 2), (86, 1), (87, 1), (88, 1), (89, 1)]

print(dictionary[66])
faith

len(dictionary)
65284
```

**2) LDA 모델 훈련**

기존의 뉴스 데이터가 총 20개의 카테고리

→ 토픽의 개수를 20으로 LDA 모델을 학습

```python
import gensim
NUM_TOPICS = 20 #20개의 토픽, k=20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, 
id2word=dictionary, passes=15)
# passes : 알고리즘 동작 횟수 <- 알고리즘이 결정하는 토픽의 값이 적절히 수렴할 수 있도록 충분히 적당한 횟수로
# num_words : 출력하고 싶은 단어의 수
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)
(0, '0.015*"drive" + 0.014*"thanks" + 0.012*"card" + 0.012*"system"')
(1, '0.009*"back" + 0.009*"like" + 0.009*"time" + 0.008*"went"')
(2, '0.012*"colorado" + 0.010*"david" + 0.006*"decenso" + 0.005*"tyre"')
(3, '0.020*"number" + 0.018*"wire" + 0.013*"bits" + 0.013*"filename"')
(4, '0.038*"space" + 0.013*"nasa" + 0.011*"research" + 0.010*"medical"')
(5, '0.014*"price" + 0.010*"sale" + 0.009*"good" + 0.008*"shipping"')
(6, '0.012*"available" + 0.009*"file" + 0.009*"information" + 0.008*"version"')
(7, '0.021*"would" + 0.013*"think" + 0.012*"people" + 0.011*"like"')
(8, '0.035*"window" + 0.021*"display" + 0.017*"widget" + 0.013*"application"')
(9, '0.012*"people" + 0.010*"jesus" + 0.007*"armenian" + 0.007*"israel"')
(10, '0.008*"government" + 0.007*"system" + 0.006*"public" + 0.006*"encryption"')
(11, '0.013*"germany" + 0.008*"sweden" + 0.008*"switzerland" + 0.007*"gaza"')
(12, '0.020*"game" + 0.018*"team" + 0.015*"games" + 0.013*"play"')
(13, '0.024*"apple" + 0.014*"water" + 0.013*"ground" + 0.011*"cable"')
(14, '0.011*"evidence" + 0.010*"believe" + 0.010*"truth" + 0.010*"church"')
(15, '0.016*"president" + 0.010*"states" + 0.007*"united" + 0.007*"year"')
(16, '0.047*"file" + 0.035*"output" + 0.033*"entry" + 0.021*"program"')
(17, '0.008*"dept" + 0.008*"devils" + 0.007*"caps" + 0.007*"john"')
(18, '0.011*"year" + 0.009*"last" + 0.007*"first" + 0.006*"runs"')
(19, '0.013*"outlets" + 0.013*"norton" + 0.012*"quantum" + 0.008*"neck"')

# 각 단어 앞에 수치는 단어의 해당 토픽에 대한 기여도
```

**3) LDA 시각화**

pyLDAvis의 설치 필요

```python
pip install pyLDAvis
```

LDA 시각화 

```python
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis)
```

**토픽 별 단어 분포**

![https://wikidocs.net/images/page/30708/visualization_final.PNG](https://wikidocs.net/images/page/30708/visualization_final.PNG)

좌측의 원들 ← 각각 20개의 토픽들

각 원과의 거리는 각 토픽들이 서로 얼마나 다른지

위의 그림에서 10번 토픽 클릭 → 우측에는 10번 토픽에 대한 정보

💡주의! LDA 모델 출력 결과 : 토픽 0~19    /  LDA 시각화 결과 : 토픽 1~20

**4) 문서 별 토픽 분포 보기**

```python
for i, topic_list in enumerate(ldamodel[corpus]):
    if i==5:
        break
    print(i,'번째 문서의 topic 비율은',topic_list)

0 번째 문서의 topic 비율은 [(7, 0.3050222), (9, 0.5070568), (11, 0.1319604), (18, 0.042834017)]
1 번째 문서의 topic 비율은 [(0, 0.031606797), (7, 0.7529218), (13, 0.02924682), (14, 0.12861845), (17, 0.037851967)]
2 번째 문서의 topic 비율은 [(7, 0.52241164), (9, 0.36602455), (16, 0.09760969)]
3 번째 문서의 topic 비율은 [(1, 0.16926806), (5, 0.04912094), (6, 0.04034211), (7, 0.11710636), (10, 0.5854137), (15, 0.02776434)]
4 번째 문서의 topic 비율은 [(7, 0.42152268), (12, 0.21917087), (17, 0.32781804)]
```

(숫자, 확률) : 각각 토픽 번호와 해당 토픽이 해당 문서에서 차지하는 분포도 의미

데이터프레임 형식으로 출력

```python
def make_topictable_per_doc(ldamodel, corpus):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼낸다
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장
            else:
                break
    return(topic_table)

topictable = make_topictable_per_doc(ldamodel, corpus)
topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해 인덱스 열을 하나 더 생성
topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
topictable[:10]
```

![https://wikidocs.net/images/page/30708/lda4.PNG](https://wikidocs.net/images/page/30708/lda4.PNG)