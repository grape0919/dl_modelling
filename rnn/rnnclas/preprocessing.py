import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from pandas.core.frame import DataFrame
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


####### SPAM DETECTION ##########
def sampleDownload():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        , filename="spam.csv")
    data = pd.read_csv('spam.csv',encoding='latin1')

    print('총 샘플 수 : ', len(data)) # 5572

    data[:5]

    # 불필요 컬럼 삭제
    del data['Unnamed: 2']
    del data['Unnamed: 3']
    del data['Unnamed: 4']

    # 라벨을 0과 1로 대체
    data['v1'] = data['v1'].replace(['ham','spam'],[0,1])

    data[:5]

    data.info()

    data.isnull().values.any()

    data.drop_duplicates(subset=['v2'], inplace=True) # 중복데이터 제거

    print("num of sample : ", len(data))

    return data

# data['v1'].value_counts().plot(kind='bar')
# plt.show()

def preproc(data:DataFrame):

    X_data = data['v2']
    Y_data = data['v1']

    tknizer = Tokenizer()
    tknizer.fit_on_texts(X_data)
    sequences = tknizer.texts_to_sequences(X_data)

    print(sequences[:5])

    word_to_index = tknizer.word_index
    print(word_to_index)

    tf_threshold = 2
    total_cnt = len(word_to_index)
    rare_cnt = 0
    total_freq = 0
    rare_freq = 0

    # 단어별 빈도수 가져온다.
    for key, value in tknizer.word_counts.items():
        # 총 빈도수
        total_freq = total_freq + value

        # 해당 단어의 빈도수가 2보다 작으면
        if(value < tf_threshold):
            rare_cnt = rare_cnt + 1 # 희귀 단어 카운트
            rare_freq = rare_freq + value # 희귀 단어에 총 빈도수

    vocab_size = len(word_to_index) + 1 # 단어 개수


    n_of_train = int(len(sequences) * 0.8) # 80% 는 학습 데이터
    n_of_test = int(len(sequences) - n_of_train) # 20%는 테스트 데이터

    # 데이터 최대 길이 
    X_data = sequences
    max_len = max(len(l) for l in X_data)

    # 패딩
    data = pad_sequences(X_data, maxlen=max_len)

    print("훈련 크기 : ", data.shape)
    print("결과 크기 : ", len(Y_data))

    X_test = data[n_of_train:]
    y_test = np.array(Y_data[n_of_train:])
    X_train = data[:n_of_train]
    y_train = np.array(Y_data[:n_of_train])

