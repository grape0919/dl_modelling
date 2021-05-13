from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

from rnn.rnngen.model import model

import os

#1) 데이터 전처리

## 단어 사전 생성
### 학습데이터
dataset_path = "/Users/hongkyokim/Documents/00.workspace/CrawlerForOCR/dataset/html"
dataset=[]
for file in os.listdir(dataset_path):

    with open(os.path.join(dataset_path, file)) as rf:
        text = rf.read()
        datas = text.split("\n")[5:]
        
        if datas:

            dataset.extend(datas)


text = """경마장에 있는 말이 뛰고 있다
그의 말이 법이다
가는 말이 고와야 오는 말이 곱다"""

### 띄어쓰기 기준 단어 분리
t = Tokenizer()
t.fit_on_texts(dataset)
#### 단어 인덱스가 1부터 시작함으로 배열의 총 크기는 +1임으로 미리 배열 크기를 늘려 설정한다.
vocab_size = len(t.word_index) + 1

### 학습데이터에 출현된 단어 리스트
print('단어 집합의 크기 : %d' % vocab_size) # 12

# print(t.word_index)

### 훈련 데이터 생성
sequences = list()
# for line in text.split('\n'):
for line in dataset:
    # 한줄씩 인코딩
    # print("line : ", line)
    encoded = t.texts_to_sequences([line])[0]
    # print("encoded[0] : ", encoded)
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print('학습에 사용할 샘플 수 : ', len(sequences))

### 레이블링
'''
예)
경마장에 있는 : 말이
경마장에 있는 말이 : 뛰고
...
'''
max_len = max(len(l) for l in sequences)
print('max lenge of sample : ', max_len)

# 샘플의 최대 길이를 확인하고
# 최대길이로 백터 생성
# 벡터 생성시 최대길이 보다 짧으면 빈공간을 0으로 채워줌 (padding='pre')
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

# print(sequences)

sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]

# print("X :", X)
# print("y : ", y)
#y :  [ 3  1  4  5  1  7  1  9 10  1 11]

y = to_categorical(y, num_classes=vocab_size)
# y 값들을 원핫 백터로 변경
# print("y : ", y)
'''
y :  [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
 '''


model(t, X, y, max_len)
