from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import os

SAVED_MODEL_FILE = "rnn/rnngen/test_model"

def model(t:Tokenizer, X, y, max_len=0):
    print("!@#!@#!@#!@#!@#!@# model ")

    if os.path.exists(SAVED_MODEL_FILE):
        model = load_model(SAVED_MODEL_FILE)
    else :
        model = Sequential()
        model.add(Embedding(len(t.word_index)+1, 10, input_length=max_len-1))
        model.add(SimpleRNN(32))
        model.add(Dense(len(t.word_index)+1, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(X, y, epochs=200, verbose=2)
        model.save(SAVED_MODEL_FILE)
    # 경마장에 라는 단어 뒤에는 4개의 단어가 있으므로 4번 예측
    print(sentence_generation(model, t, '가는', 5))


def sentence_generation(model: Sequential, t: Tokenizer, current_word, n):
    init_word = current_word # 처음 들어온 단어 저장
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0] # 단어사전의 인덱스(정수)로 인코딩
        # print("test to seq : ", encoded)
        encoded = pad_sequences([encoded], maxlen=5, padding='pre') # 패딩 (5 크기로 앞쪽에 0으로 채움)
        # print("padded encoded : ", encoded)
        result = model.predict_classes(encoded, verbose=0)
          
        for word, index in t.word_index.items():
            if index == result:
                break
        
        current_word = current_word + ' ' + word
        sentence = sentence + ' ' + word
        
    sentence = init_word + sentence

    return sentence

