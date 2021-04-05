# 딥 러닝을 이용한 자연어 처리 입문     

​      

## Text Generation using RNN      

<br/>

Example)       

'경마장에 있는 말이 뛰고 있다'      

 '그의 말이 법이다'       

 '가는 말이 고와야 오는 말이 곱다'       

전체 문장의 앞의 단어들을 전부 고려하여 학습하도록 데이터를 재구성한다면        

->

| samples | X                          | y      |
| ------- | -------------------------- | ------ |
| 1       | 경마장에                   | 있는   |
| 2       | 경마장에 있는              | 말이   |
| 3       | 경마장에 있는 말이         | 뛰고   |
| 4       | 경마장에 있는 말이 뛰고    | 있다   |
| 5       | 그의                       | 말이   |
| 6       | 그의 말이                  | 법이다 |
| 7       | 가는                       | 말이   |
| 8       | 가는 말이                  | 고와야 |
| 9       | 가는 말이 고와야           | 오는   |
| 10      | 가는 말이 고와야 오는      | 말이   |
| 11      | 가는 말이 고와야 오는 말이 | 곱다   |

​         

​       

### **LSTM을 이용하여 텍스트 생성하기**      

​      

사용할 데이터는 __뉴욕 타임즈 기사의 제목__       

[ArticlesApril2018.csv](https://www.kaggle.com/aashita/nyt-comments)             

```python
import pandas as pd
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical  

df=pd.read_csv('ArticlesApril2018.csv')

print('열의 개수: ',len(df.columns))
print(df.columns)

열의 개수:  15
Index(['articleID', 'articleWordCount', 'byline', 'documentType', 'headline',
       'keywords', 'multimedia', 'newDesk', 'printPage', 'pubDate',
       'sectionName', 'snippet', 'source', 'typeOfMaterial', 'webURL'], dtype='object')

df['headline'].isnull().values.any()
False
```

​       

headline 열에서 모든 신문 기사의 제목을 뽑아서 하나의 리스트로 저장

```python
headline = [] # 리스트 선언
headline.extend(list(df.headline.values)) # 헤드라인의 값들을 리스트로 저장
headline[:5] # 상위 5개만 출력

['Former N.F.L. Cheerleaders’ Settlement Offer: $1 and a Meeting With Goodell',
 'E.P.A. to Unveil a New Rule. Its Effect: Less Science in Policymaking.',
 'The New Noma, Explained',
 'Unknown',
 'Unknown']
```

​          

'Unknown'  <-  노이즈 데이터 제거해줄 필요    

```python
print('총 샘플의 개수 : {}'.format(len(headline))) # 현재 샘플의 개수
총 샘플의 개수 : 1324

headline = [n for n in headline if n != "Unknown"] # Unknown 값을 가진 샘플 제거
print('노이즈값 제거 후 샘플의 개수 : {}'.format(len(headline))) # 제거 후 샘플의 개수
노이즈값 제거 후 샘플의 개수 : 1214
    
headline[:5]
['Former N.F.L. Cheerleaders’ Settlement Offer: $1 and a Meeting With Goodell',
 'E.P.A. to Unveil a New Rule. Its Effect: Less Science in Policymaking.',
 'The New Noma, Explained',
 'How a Bag of Texas Dirt  Became a Times Tradition',
 'Is School a Place for Self-Expression?']
```

​         

__데이터 전처리__ 수행 (__구두점 제거__ , __소문자화__ )      

```python
def repreprocessing(s):
    s=s.encode("utf8").decode("ascii",'ignore')
    return ''.join(c for c in s if c not in punctuation).lower() # 구두점 제거와 동시에 소문자화

text = [repreprocessing(x) for x in headline]
text[:5]
['former nfl cheerleaders settlement offer 1 and a meeting with goodell',
 'epa to unveil a new rule its effect less science in policymaking',
 'the new noma explained',
 'how a bag of texas dirt  became a times tradition',
 'is school a place for selfexpression']
```

​        

__토크나이즈 __   

```python
t = Tokenizer()
t.fit_on_texts(text)
vocab_size = len(t.word_index) + 1

print('단어 집합의 크기 : %d' % vocab_size)
단어 집합의 크기 : 3494
```

​        

정수 인코딩과 동시에 하나의 문장을 여러 줄로 분해하여 훈련 데이터를 구성  

```python
sequences = list()

for line in text: # 1,214 개의 샘플에 대해서 샘플을 1개씩 
    encoded = t.texts_to_sequences([line])[0] # 각 샘플에 대한 정수 인코딩
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

sequences[:11] # 11개의 샘플 출력
[[99, 269], # former nfl
 [99, 269, 371], # former nfl cheerleaders
 [99, 269, 371, 1115], # former nfl cheerleaders settlement
 [99, 269, 371, 1115, 582], # former nfl cheerleaders settlement offer
 [99, 269, 371, 1115, 582, 52], # 'former nfl cheerleaders settlement offer 1
 [99, 269, 371, 1115, 582, 52, 7], # former nfl cheerleaders settlement offer 1 and
 [99, 269, 371, 1115, 582, 52, 7, 2], 
 [99, 269, 371, 1115, 582, 52, 7, 2, 372],
 [99, 269, 371, 1115, 582, 52, 7, 2, 372, 10],
 [99, 269, 371, 1115, 582, 52, 7, 2, 372, 10, 1116], # 모든 단어 사용된 완전한 첫번째 문장
 # 바로 위의 줄 former nfl cheerleaders settlement offer 1 and a meeting with goodell
 [100, 3]] # epa to에 해당되며 두번째 문장 시작
```

​       

__패딩__   

```python
max_len=max(len(l) for l in sequences)
print('샘플의 최대 길이 : {}'.format(max_len))
샘플의 최대 길이 : 24

sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences[:3])
[[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0    0   99  269]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0   99  269  371]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   99  269  371 1115]
```

​         

```python
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
print(X[:3])

[[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0    0   99]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0   99  269]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   99  269  371]
```

​        

```python
print(y[:3]) # 레이블
[ 269  371 1115]
y = to_categorical(y, num_classes=vocab_size)
```

   

<br/>

<br/>

__모델 설계__       

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
# y데이터 분리했으므로 X데이터의 길이는 기존 데이터의 길이 - 1
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)
```

​       

​       

문장 생성하는 함수 sentence_generation으로 문장 생성

```python
def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n): # n번 반복
        encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=23, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)
    # 입력한 X(현재 단어)에 대해 y 예측하고 y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items(): 
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break 
        current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word # 예측 단어 문장에 저장
    
    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, t, 'i', 10))
# 'i'에 대해 10개 단어 추가 생성
i disapprove of school vouchers can i still apply for them

print(sentence_generation(model, t, 'how', 10))
# 'how'에 대해 10개 단어 추가 생성
how to make facebook more accountable will so your neighbor chasing
```

