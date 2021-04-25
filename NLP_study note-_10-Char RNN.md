# 딥 러닝을 이용한 자연어 처리 입문      

​         

## 글자 단위 RNN (Char RNN)        

​      

입출력의 단위를 단어 레벨(word-level)에서 글자 레벨(character-level)로 변경

![image](https://wikidocs.net/images/page/48649/char_rnn1.PNG)

<br/>

### 1. 글자 단위 RNN 언어 모델(Char RNNLM)      

글자 단위를 입, 출력으로 사용하므로 임베딩층(embedding layer)을 사용하지 않는다      

[이상한 나라의 앨리스(Alice’s Adventures in Wonderland)](http://www.gutenberg.org/files/11/11-0.txt)

​       

```python
import numpy as np
import urllib.request
from tensorflow.keras.utils import to_categorical
urllib.request.urlretrieve("http://www.gutenberg.org/files/11/11-0.txt", filename="11-0.txt")
f = open('11-0.txt', 'rb')
lines=[]
for line in f: 
    line=line.strip() # \r, \n 제거
    line=line.lower() 
    line=line.decode('ascii', 'ignore') # \xe2\x80\x99 등과 같은 바이트 열 제거
    if len(line) > 0:
        lines.append(line)
f.close()

lines[:5]
['project gutenbergs alices adventures in wonderland, by lewis carroll',
 'this ebook is for the use of anyone anywhere at no cost and with',
 'almost no restrictions whatsoever.  you may copy it, give it away or',
 're-use it under the terms of the project gutenberg license included',
 'with this ebook or online at www.gutenberg.org']
```

​       

이를 하나의 문자열로 통합

```python
text = ' '.join(lines)
print('문자열의 길이 또는 총 글자의 개수: %d' % len(text))
문자열의 길이 또는 총 글자의 개수: 158783

print(text[:200])
project gutenbergs alices adventures in wonderland, by lewis carroll this ebook is for the use of anyone anywhere at no cost and with almost no restrictions whatsoever.  you may copy it, give it away 
```

​       

이 문자열로부터 글자 집합

```python
char_vocab = sorted(list(set(text)))
vocab_size=len(char_vocab)
print ('글자 집합의 크기 : {}'.format(vocab_size))
글자 집합의 크기 : 55
    
char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) 
print(char_to_index)
{' ': 0, '!': 1, '#': 2, '$': 3, '%': 4, '(': 5, ')': 6, '*': 7, ',': 8, '-': 9, '.': 10, '/': 11, '0': 12, '1': 13, '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20, '9': 21, ':': 22, ';': 23, '?': 24, '@': 25, '[': 26, ']': 27, '_': 28, 'a': 29, 'b': 30, 'c': 31, 'd': 32, 'e': 33, 'f': 34, 'g': 35, 'h': 36, 'i': 37, 'j': 38, 'k': 39, 'l': 40, 'm': 41, 'n': 42, 'o': 43, 'p': 44, 'q': 45, 'r': 46, 's': 47, 't': 48, 'u': 49, 'v': 50, 'w': 51, 'x': 52, 'y': 53, 'z': 54}
```

​       

인덱스로부터 글자를 리턴하는 index_to_char

```python
index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value] = key
```

​       

훈련 데이터 구성      

```python
# Example) 샘플의 길이가 4라면 4개의 입력 글자 시퀀스로 부터 4개의 출력 글자 시퀀스를 예측. 즉, RNN의 time step은 4번
appl -> pple
# appl은 train_X, pple는 train_y에 저장
```

​       

text 문자열로부터 다수의 문장 샘플들로 분리

```python
seq_length = 60 # 문장의 길이 60
n_samples = int(np.floor((len(text) - 1) / seq_length)) 
print ('문장 샘플의 수 : {}'.format(n_samples))
문장 샘플의 수 : 2646
```

​       

```python
train_X = []
train_y = []

for i in range(n_samples): # 2,646번 수행
    X_sample = text[i * seq_length: (i + 1) * seq_length]
    # 0:60 -> 60:120 -> 120:180로 loop를 돌면서 문장 샘플 1개씩 
    X_encoded = [char_to_index[c] for c in X_sample] # 하나의 문장 샘플에 대해 정수 인코딩
    train_X.append(X_encoded)

    y_sample = text[i * seq_length + 1: (i + 1) * seq_length + 1] # 오른쪽으로 1칸 쉬프트
    y_encoded = [char_to_index[c] for c in y_sample]
    train_y.append(y_encoded)
    
print(train_X[0])
[44, 46, 43, 38, 33, 31, 48, 0, 35, 49, 48, 33, 42, 30, 33, 46, 35, 47, 0, 29, 40, 37, 31, 33, 47, 0, 29, 32, 50, 33, 42, 48, 49, 46, 33, 47, 0, 37, 42, 0, 51, 43, 42, 32, 33, 46, 40, 29, 42, 32, 8, 0, 30, 53, 0, 40, 33, 51, 37, 47]
print(train_y[0])
[46, 43, 38, 33, 31, 48, 0, 35, 49, 48, 33, 42, 30, 33, 46, 35, 47, 0, 29, 40, 37, 31, 33, 47, 0, 29, 32, 50, 33, 42, 48, 49, 46, 33, 47, 0, 37, 42, 0, 51, 43, 42, 32, 33, 46, 40, 29, 42, 32, 8, 0, 30, 53, 0, 40, 33, 51, 37, 47, 0]
```

​       

```python
train_X = to_categorical(train_X)
train_y = to_categorical(train_y)
print('train_X의 크기(shape) : {}'.format(train_X.shape)) # 원-핫 인코딩
print('train_y의 크기(shape) : {}'.format(train_y.shape)) # 원-핫 인코딩
train_X의 크기(shape) : (2646, 60, 55)
train_y의 크기(shape) : (2646, 60, 55)
```

![image](https://wikidocs.net/images/page/22886/rnn_image6between7.PNG)

​         

<br/>

모델 설계   

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed

model = Sequential()
model.add(LSTM(256, input_shape=(None, train_X.shape[2]), return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=80, verbose=2)


def sentence_generation(model, length):
    ix = [np.random.randint(vocab_size)] # 글자에 랜덤 인덱스 생성
    y_char = [index_to_char[ix[-1]]] # 랜덤 인덱스로 글자 생성
    print(ix[-1],'번 글자',y_char[-1],'로 예측을 시작!')
    X = np.zeros((1, length, vocab_size)) # (1, length, 55) 크기의 X 생성. 즉, LSTM의 입력 시퀀스 생성

    for i in range(length):
        X[0][i][ix[-1]] = 1 # X[0][i][예측한 글자의 인덱스] = 1, 즉, 예측 글자를 다음 입력 시퀀스에 추가
        print(index_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(index_to_char[ix[-1]])
    return ('').join(y_char)
sentence_generation(model, 100)


49 번 글자 u 로 예측을 시작!
ury-men would have done just as well. the twelve jurors were to say in that dide. he went on in a di'
```

​         

<br/>

<br/>

### 2. 글자 단위 RNN(Char RNN)으로 텍스트 생성하기      

```python
import numpy as np
from tensorflow.utils import to_categorical
# 임의의 노래 가사

text='''
I get on with life as a programmer,
I like to contemplate beer.
But when I start to daydream,
My mind turns straight to wine.

Do I love wine more than beer?

I like to use words about beer.
But when I stop my talking,
My mind turns straight to wine.

I hate bugs and errors.
But I just think back to wine,
And I'm happy once again.

I like to hang out with programming and deep learning.
But when left alone,
My mind turns straight to wine.
'''

tokens = text.split() # '\n 제거'
text = ' '.join(tokens)
print(text)

I get on with life as a programmer, I like to contemplate beer. But when I start to daydream, My mind turns straight to wine. Do I love wine more than beer? I like to use words about beer. But when I stop my talking, My mind turns straight to wine. I hate bugs and errors. But I just think back to wine, And I'm happy once again. I like to hang out with programming and deep learning. But when left alone, My mind turns straight to wine.

char_vocab = sorted(list(set(text))) # 중복을 제거한 글자 집합 생성
print(char_vocab)
[' ', "'", ',', '.', '?', 'A', 'B', 'D', 'I', 'M', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y']
```

​         

글자 집합      

```python
vocab_size=len(char_vocab)
print ('글자 집합의 크기 : {}'.format(vocab_size))
글자 집합의 크기 : 33

char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 글자에 고유한 정수 인덱스 부여
print(char_to_index)
{' ': 0, "'": 1, ',': 2, '.': 3, '?': 4, 'A': 5, 'B': 6, 'D': 7, 'I': 8, 'M': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'r': 26, 's': 27, 't': 28, 'u': 29, 'v': 30, 'w': 31, 'y': 32}
```

​       

훈련에 사용할 문장 샘플들

```python
# Example) 5개의 입력 글자 시퀀스로부터 다음 글자 시퀀스 예측. 즉, RNN의 time step은 5번
stude -> n 
tuden -> t
```

​      

입력 시퀀스의 길이. 즉, 모든 샘플들의 길이가 10가 되도록

```python
length = 11
sequences = []
for i in range(length, len(text)):
    seq = text[i-length:i] # 길이 11의 문자열
    sequences.append(seq)
print('총 훈련 샘플의 수: %d' % len(sequences))
총 훈련 샘플의 수: 426


sequences[:10]
['I get on wi',
 ' get on wit',
 'get on with',
 'et on with ',
 't on with l',
 ' on with li',
 'on with lif',
 'n with life',
 ' with life ',
 'with life a']
```

​        

정수 인코딩

```python
X = []
for line in sequences: # 문장 샘플을 1개씩 
    temp_X = [char_to_index[char] for char in line] # 각 글자에 대해 정수 인코딩
    X.append(temp_X)


for line in X[:5]:
    print(line)
[8, 0, 16, 14, 28, 0, 24, 23, 0, 31, 18]
[0, 16, 14, 28, 0, 24, 23, 0, 31, 18, 28]
[16, 14, 28, 0, 24, 23, 0, 31, 18, 28, 17]
[14, 28, 0, 24, 23, 0, 31, 18, 28, 17, 0]
[28, 0, 24, 23, 0, 31, 18, 28, 17, 0, 21]
```

​        

```python
sequences = np.array(X)
X = sequences[:,:-1]
y = sequences[:,-1] 

for line in X[:5]:
    print(line)
[ 8  0 16 14 28  0 24 23  0 31]
[ 0 16 14 28  0 24 23  0 31 18]
[16 14 28  0 24 23  0 31 18 28]
[14 28  0 24 23  0 31 18 28 17]
[28  0 24 23  0 31 18 28 17  0]
print(y[:5])
[18 28 17  0 21]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X] # X 원-핫 인코딩
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size) # y 원-핫 인코딩

print(X.shape)
(426, 10, 33)
```

![image](https://wikidocs.net/images/page/22886/rnn_image6between7.PNG)

​       

모델

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = Sequential()
model.add(LSTM(80, input_shape=(X.shape[1], X.shape[2]))) # X.shape[1]은 25, X.shape[2]는 33
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=2)

def sentence_generation(model, char_to_index, seq_length, seed_text, n):
    init_text = seed_text # 문장 생성에 사용할 초기 시퀀스
    sentence = ''

    for _ in range(n): 
        encoded = [char_to_index[char] for char in seed_text] # 현재 시퀀스에 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre') 
        encoded = to_categorical(encoded, num_classes=len(char_to_index))
        result = model.predict_classes(encoded, verbose=0)
       
        for char, index in char_to_index.items(): 
            if index == result: 
                break
        seed_text=seed_text + char 
        sentence=sentence + char

    sentence = init_text + sentence
    return sentence

print(sentence_generation(model, char_to_index, 10, 'I get on w', 80))
I get on with life as a programmer, I like to hang out with programming and deep learning.
```

