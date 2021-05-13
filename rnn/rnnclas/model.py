from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


SAVED_MODEL_FILEPATH = "rnn/rnnclas/test_model"

def model(X_train, y_train, vocab_size:int=0):

    model = Sequential()
    model.add(Embedding(vocab_size, 32)) # 32차원의 임배딩
    model.add(SimpleRNN(32)) # hidden layouts = 32
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss="binary_crossentropy", metrics=['acc'])
    history  = model.fit(X_train, y_train, epochs=4, batch_size=64, validation_split=0.2)

    return model

def saveModel(model:Sequential):
    model.save(SAVED_MODEL_FILEPATH)
    return model

def loadModel():
    model = load_model(SAVED_MODEL_FILEPATH)
    return model

def eval(X_test, y_test, model:Sequential):
    return model.evaluate(X_test, y_test)[1]