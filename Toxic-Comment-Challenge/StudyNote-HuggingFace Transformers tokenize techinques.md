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

### __단어 단위 토크나이즈__는        

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

“mug”는 “m”이 사전에 등록돼 있지 않으므로 ['<unk>', 'ug']로 분절이 된다      

​	_일반적으로 a-z와 같은 기본 캐릭터들은 기본 사전에 미리 등록되기 때문에 이러한 현상이 “m”에 대해 발생하지는 않겠지만, 이모지와 같은 스페셜 캐릭터들에 대해서는 위 같은 <unk> 치환이 많이 발생하게 된다_          

<br/>

__사전의 크기 (일반적으로 기본 단어 개수 + 합쳐진 서브 워드의 개수)__  는  사용자가 정하는 __하이퍼파라미터__        

예를 들어 [GPT](https://huggingface.co/transformers/master/model_doc/gpt.html)의 경우, 478개의 기본 캐릭터에 40,000 개의 서브 워드를 더하도록 설정했기 때문에 총 40,478의 사전 크기를 지닌다  

<br/>

<br/>

