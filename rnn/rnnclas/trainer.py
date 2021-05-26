

import rnn.rnnclas.model as model
import rnn.rnnclas.preprocessing as preproc
from matplotlib import pyplot as plt

def train():
    print("## Loading dataset...")
    data = preproc.loadDataset()

    print("## Preprocessing...")
    x_train, y_train, x_test, y_test, vocab_size = preproc.preproc(data)

    print("## Training... ")
    m, h = model.model(x_train,y_train, vocab_size)

    print("## Show result for training")
    # epochs = range(1, len(h.history['acc']) + 1)
    # plt.plot(epochs, h.history['loss'])
    # plt.plot(epochs, h.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    model.saveModel(m)

    print(f"테스트 정확도 : {model.eval(x_test,y_test,m)}%")

    test_result = m.predict_classes(x_test)
    print(test_result)

def loadModel():
    return model.loadModel()

if __name__=="__main__":

    train()

