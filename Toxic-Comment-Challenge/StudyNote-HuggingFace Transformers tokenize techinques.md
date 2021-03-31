# HuggingFace 내 토크나이저 종류 살펴보기      

​      

Transformers 라이브러리 내 토크나이저 종류를 살펴보자     

<br/>

------

### Transformers 라이브러리에서 활용되는 세 가지 핵심 토크나이즈 기법:      

### [Byte-Pair Encoding (BPE)](https://huggingface.co/transformers/master/tokenizer_summary.html#byte-pair-encoding)      

### [WordPiece](https://huggingface.co/transformers/master/tokenizer_summary.html#wordpiece)     

### [SentencePiece](https://huggingface.co/transformers/master/tokenizer_summary.html#sentencepiece) 

<br/>

<br/>

-----

### Tokenize?       

입력 문장을 단어 혹은 서브 워드 단위로 쪼갠 후, 사전에 등록된 아이디로 변환해주는 과정        

<br/>

------

문장을 토큰으로 분절할 때 사용하는 규칙에 따라, 우리는 동일한 문장에 대해서도 서로 다른 토큰 리스트를 반환 받을 수 있다     

Example)  “Don’t you love Transoformers? We sure do.”

```python
["Don't", "you", "love", "Transformers?", "We", "sure", "do."]

["Don", "'", "t", "you", "love", "Transformers", "?", "We", "sure", "do", "."]
```

​       

[spaCy](https://spacy.io/)와 [Moses](http://www.statmt.org/moses/?n=Development.GetStarted)는 유명한 규칙 기반 토크나이저 라이브러리

이 두 토크나이저를 이용하면      

```python
["Do", "n't", "you", "love", "Transformers", "?", "We", "sure", "do", "."]
```

<br/>

<br/>

------------

### __단어 단위 토크나이즈__ 는        

문장을 분절하는 가장 직관적인 방법이지만, 해당 기법은 코퍼스 크기에 따라 엄청나게 큰 사전을 사용해야 할 수 있다는 단점      

-> [TransformerXL ](https://huggingface.co/transformers/master/model_doc/transformerxl.html)은 공백/구두점을 기준으로 분절을 수행하였기 때문에 총 267,735 개의 토큰을 지니는 사전을 활용   -> 메모리 문제 야기  

<br/>

### 캐릭터 기반 토크나이즈는        

매우 단순하고, 메모리를 엄청나게 절약할 수 있다는 장점이 있지만       

이는 모델이 텍스트 표현을 의미있는 단위로 학습하는데 지장을 주게 되고 결과적으로 성능의 감소를 가져온다     

 <br/>

### 따라서 두 세계의 중간에 위치한 서브 워드 단위의 분절을 활용하는 것이 바람직      

<br/>

----------------

### 서브 워드 기반 토크나이즈

“__자주 등장한 단어는 그대로 두고, 자주 등장하지 않은 단어는 의미있는 서브 워드 토큰들로 분절한다__ ”

Example) 'annoyingly'  ->  'annoying'  +  'ly'         

__장점__      

의미있는 단어 혹은 서브 워드 단위의 표현을 학습하면서도, 합리적인 사전 크기를 유지할 수 있다는 점      

훈련 과정에서 만나지 않은 단어에 대해서도 유의미한 분절을 수행할 수 있다

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# uncased 모델을 활용 -> 문장 자체 소문자로 대체
tokenizer.tokenize("I have a new GPU!")

>>> ["i", "have", "a", "new", "gp", "##u", "!"]
# “##”는 해당 심볼을 지닌 토큰은 해당 토큰 이전에 등장한 토큰과 공백 없이 합쳐져야 한다는 의미
```

<br/>

```python
from transformers import XLNetTokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokenizer.tokenize("Don't you love transformers? We sure do.")
>>> ["▁Don", "'", "t", "▁you", "▁love", "▁", "Transform", "ers", "?", "▁We", "▁sure", "▁do", "."]

```

<br/>

-------

### 서브 워드 토크나이즈 알고리즘        

<br/>

#### Byte-Pair Encoding (BPE)      

일반적으로 훈련 데이터를 단어 단위로 분절하는 Pre-tokenize 과정을 거쳐야 한다     

이러한 Pre-tokenize는 단순한 공백 단위로 수행될 수 있고, 앞서 언급한 규칙 기반의 토크나이즈를 거쳐 수행될 수 있다

<br/>

Example)

Pre-tokenize를 거친 후

```python
('hug', 10), ('pug', 5), ('pun', 12), ('bun', 4), ('hugs', 5)
# 정수 값  :  각 단어가 얼마나 등장했는지 나타내는 값
```

​     

__기본 사전은   ['b', 'g', 'h', 'n', 'p', 's', 'u']__       

본 사전을 기반으로 위에서 얻어진 단어들을 캐릭터 단위로 쪼개면      

```python
('h' 'u' 'g', 10), ('p' 'u' 'g', 5), ('p' 'u' 'n', 12), ('b' 'u' 'n', 4), ('h' 'u' 'g' 's', 5)
```

​        

함께 가장 많이 등장한 캐릭터 쌍 확인      

 “hu” 총 15번,  “ug” 총 20번이 나와 가장 많이 등장한 쌍은 “ug”      

따라서 “u”와 “g”를 합친 “ug”를 사전에 새로 추가

```python
('h' 'ug', 10), ('p' 'ug', 5), ('p' 'u' 'n', 12), ('b' 'u' 'n', 4), ('h' 'ug' 's', 5)
```

​            

다음에 가장 많이 나온 쌍은 16번 등장한 “un”이므로, “un” 사전에 추가      

그 다음은 15번 등장한 “hug”이므로 “hug”도 사전에 추가       

이 시점에서의 사전은 ['b', 'g', 'h', n', 'p', 's', 'u', 'ug', 'un', 'hug']      

```python
('hug', 10), ('p' 'ug', 5), ('p' 'un', 12), ('b' 'un', 4), ('hug' 's', 5)
```

​       

여기서 훈련을 멈추게 되면, 토크나이저는 지금까지 배운 규칙을 가지고 새로운 단어들을 분절하게 된다      

예를 들어, “bug”는 ['b', 'ug']로 분절,       

“mug”는 “m”이 사전에 등록돼 있지 않으므로 ['< unk >', 'ug']로 분절이 된다      

​	_일반적으로 a-z와 같은 기본 캐릭터들은 기본 사전에 미리 등록되기 때문에 이러한 현상이 “m”에 대해 발생하지는 않겠지만, 이모지와 같은 스페셜 캐릭터들에 대해서는 위 같은 < unk > 치환이 많이 발생하게 된다_          

<br/>

__사전의 크기 (일반적으로 기본 단어 개수 + 합쳐진 서브 워드의 개수)__  는  사용자가 정하는 __하이퍼파라미터__        

예를 들어 [GPT](https://huggingface.co/transformers/master/model_doc/gpt.html)의 경우, 478개의 기본 캐릭터에 40,000 개의 서브 워드를 더하도록 설정했기 때문에 총 40,478의 사전 크기를 지닌다  

<br/>

<br/>

-------

#### Byte-level BPE      

사전이 모든 유니코드 캐릭터들을 지니게 하기 위해서는 꽤나 큰 크기로 사전을 지정해야 한다      

그러나 [GPT-2 논문](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)에서는 바이트를 사전의 기본 단위로 사용하고 구두점을 처리하기 위한 몇 가지 추가 규칙을 더해, Byte-level BPE는 `<unk>`으로의 치환 없이 모든 텍스트를 분절할 수 있게 되었다     

따라서 [GPT-2 모델](https://huggingface.co/transformers/master/model_doc/gpt.html)은 256개의 기본 바이트 토큰과 `<end-of-text>` 토큰 그리고 50,000 개의 서브 워드를 더해 총 50,257 개의 토큰 사전을 지닌다     

<br/>

<br/>

--------

#### WordPiece      

[BERT](https://huggingface.co/transformers/master/model_doc/bert.html)에서 활용된 서브 워드 토크나이즈 알고리즘        

 BPE와 마찬가지로 사전을 코퍼스 내 등장한 캐릭터들로 초기화 한 후, 사용자가 지정한 횟수 만큼 서브 워드를 병합하는 방식으로 훈련     

__한 가지 다른 점이 있다면__ WordPiece는 BPE와 같이 가장 많이 등장한 쌍을 병합하는 것이 아니라, 병합되었을 때 코퍼스의 Likelihood를 가장 높이는 쌍을 병합하게 된다

코퍼스 내에서 “ug”가 등장할 확률을 “u”와 “g”가 각각 등장할 확률을 곱한 값으로 나눈 값이 다른 쌍보다 클 경우 해당 쌍을 병합      

병합 후보에 오른 쌍을 미리 병합해보고 잃게 되는 것은 무엇인지, 해당 쌍을 병합할 가치가 충분한지 등을 판단한 후에 병합을 수행한다는 점에 있어 BPE와 다르다고 할 수 있다      

<br/>

<br/>

----------------

#### Unigram

BPE, WordPiece와 같이 기본 캐릭터에서 서브 워드를 점진적으로 병합해나가는 것이 아니라,      

모든 Pre-tokenized 토큰과 서브 워드에서 시작해 점차 사전을 줄여나가는 방식으로 진행      

Unigram은 [SentencePiece](https://huggingface.co/transformers/master/tokenizer_summary.html#sentencepiece)에서 주로 활용이 되는 알고리즘      

​      

매 스텝마다 Unigram은 주어진 코퍼스와 현재 사전에 대한 Loss를 측정      

이후, 각각의 서브 워드에 대해 해당 서브 워드가 코퍼스에서 제거되었을 때, Loss가 얼마나 증가하는지를 측정      

이에 따라 Loss를 가장 조금 증가시키는 p(보통 전체 사전 크기의 10-20% 값으로 설정)개 토큰을 제거       

Unigram은 해당 과정을 사용자가 원하는 사전 크기를 지니게 될 때 까지 반복       

기본 캐릭터들은 반드시 사전에서 제거되지 않고 유지

<br/>

<br/>

-------

#### SentencePiece      

지금까지 우리가 살펴본 모든 방법들은 Pre-tokenize 과정을 필요로 하며,  이는 아주 중요한 문제를 야기     <- 모든 언어가 공백을 기준으로 단어를 분절할 수 없기 때문     

SentencePiece는 입력 문장을 Raw Stream으로 취급해 공백을 포함한 모든 캐릭터를 활용해, BPE 혹은 Unigram을 적용하며 사전을 구축

  	<- 이것이 바로 앞선 `XLNetTokenizer`의 예에서 공백을 나타내는 “▁” 캐릭터를 만나게 된 이유

  	<- 단순히 모든 토큰들을 붙여준 후, “▁” 캐릭터만 공백으로 바꿔주면 되기 때문에 분절된 텍스트를 디코드하는 작업은 매우 쉽다      

​      

Transformers 라이브러리가 지원하는 모델들 중 SentencePiece를 활용하는 모든 모델들의 토크나이저는 Unigram을 활용해 훈련되었다     

Example) ALBERT, XLNet, Marian NMT 