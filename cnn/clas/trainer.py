

import cnn.clas.model as model
import cnn.clas.preprocessing as proc


if __name__ == '__main__':
    print("!@#!@# start trainer")
    data = proc.loadDataset()

    print("!@#!@# data : ", len(data))
    x_train, y_train, x_test, y_test, vocab_size = proc.preproc(data)

    print("!@#!@# end preproc : ", x_train[0])
    m, h = model.model(x_train, y_train, vocab_size)

    model.eval(m, x_test, y_test)

