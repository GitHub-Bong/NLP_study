# 딥 러닝을 이용한 자연어 처리 입문     

​       

## FastText     

단어를 벡터로 만드는 또 다른 방법으로 페이스북에서 개발한 FastText가 있다      

Word2Vec 이후에 나온 것이기 때문에, 메커니즘 자체는 Word2Vec의 확장이라고 볼 수 있다      

​     

Word2Vec와 FastText와의 가장 큰 차이점이라면 __Word2Vec__ 는 __단어를 쪼개질 수 없는 단위로 생각한다면__ ,        

__FastText__ 는 __하나의 단어 안에도 여러 단어들이 존재하는 것__으로 간주 즉 __내부 단어(subword)__ 를 고려하여 학습한다       

<br/>

## 내부 단어 (subword)의 학습      

각 단어는 글자 단위 n-gram의 구성으로 취급

Example)      

```python
# n = 3인 경우
<ap, app, ppl, ple, le>, <apple>
```

​      

실제 사용할 때는 n의 최소값과 최대값으로 범위를 설정할 수 있는데, 기본값으로 각각 3과 6으로 설정되어 있다     

Example) 최소값 = 3, 최대값 = 6인 경우     

단어 apple에 대해서 FastText는 

```python
# n = 3 ~ 6인 경우
<ap, app, ppl, ppl, le>, <app, appl, pple, ple>, <appl, pple>, ..., <apple>
```

이렇게 내부 단어 벡터화     

내부 단어들을 벡터화한다는 의미  ->  저 단어들에 대해서 Word2Vec을 수행한다는 의미      

단어 apple의 벡터값은 저 위 벡터값들의 총 합으로 구성     

```python
apple = <ap + app + ppl + ppl + le> + <app + appl + pple + ple> + <appl + pple> + , ..., +<apple>
```

이런 방법은 Word2Vec에서는 얻을 수 없었던 강점을 가진다 

<br/>

<br/>

## 모르는 단어 (Out of Vocabulary, OOV) 에 대한 대응      

장점    

데이터 셋만 충분한다면 위와 같은 내부 단어(Subword)를 통해 모르는 단어(Out Of Vocabulary, OOV)에 대해서도 다른 단어와의 유사도를 계산할 수 있다     

​     

Example)     

FastText에서 birthplace(출생지)란 단어를 학습하지 않은 상태라고 가정     

하지만 다른 단어에서 birth와 place라는 내부 단어가 있었다면,        

FastText는 birthplace의 벡터를 얻을 수 있다        

-> 이는 모르는 단어에 제대로 대처할 수 없는 Word2Vec, GloVe와는 다른 점

<br/>

<br/>

## 단어 집합 내 빈도 수가 적었던 단어(Rare Word)에 대한 대응      

Word2Vec의 경우 등장 빈도 수가 적은 단어(rare word)에 대해서 참고할 수 있는 경우의 수가 적다보니 임베딩의 정확도가 높지 않다는 단점이 있다     

FastText의 경우, 만약 단어가 희귀 단어라도, 그 단어의 n-gram이 다른 단어의 n-gram과 겹치는 경우라면, Word2Vec과 비교하여 비교적 높은 임베딩 벡터값을 얻는다     

​      

FastText가 노이즈가 많은 코퍼스에서 강점을 가진 것 또한 이와 같은 이유      

실제 많은 비정형 데이터에는 오타(typo)가 섞여있다     

오타가 섞인 단어는 당연히 등장 빈도수가 매우 적으므로 일종의 희귀 단어     

FastText는 이에 대해서도 일정 수준의 성능을 보인다      

<br/>

<br/>

## 한국어에서의 FastText     

한국어의 경우에도 OOV 문제를 해결하기 위해 FastText를 적용하고자 하는 시도들이 있다     

__(1) 음절 단위__     

```python
# n=3 일 때
<자연, 자연어, 연어처, 어처리, 처리>
```

__(2) 자모 단위__  

오타나 노이즈 측면에서 더 강한 임베딩을 기대해볼 수 있다      

만약, 종성이 존재하지 않는다면 ‘_’라는 토큰을 사용한다고 가정     

```python
# ‘자연어처리’라는 단어는 아래와 같이 분리가 가능
분리된 결과 : ㅈ ㅏ _ ㅇ ㅕ ㄴ ㅇ ㅓ _ ㅊ ㅓ _ ㄹ ㅣ _ 
# n=3일 때
< ㅈ ㅏ, ㅈ ㅏ _, ㅏ _ ㅇ, ... 중략>
```

