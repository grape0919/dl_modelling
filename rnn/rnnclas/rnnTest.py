import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


####### SPAM DETECTION ##########
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