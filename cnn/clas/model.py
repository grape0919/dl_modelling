
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Layer, Conv1D, Embedding, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def model(x_train, y_train, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 32))
    model.add(Dropout(0.2))
    model.add(Conv1D(32, 5, strides=1, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose=1, patience=3)
    mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode='max', verbose=1, save_best_only=True)

    history = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2, callbacks=[es, mc])

    return model, history

def eval(model:Sequential, x_test, y_test):
    print("테스트 정확도 : ", model.evaluate(x_test, y_test)[1])
    