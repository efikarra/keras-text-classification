from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
from keras.utils.np_utils import to_categorical
import model

vocab_size = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

# print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# x_train = np.random.rand(100, 20).astype('float32')
# x_test = np.random.rand(100, 20).astype('float32')
# # input[:,19,:,0]=0.0
# #     print input
# print "x_train ", x_train.shape
# # 10 conversations with 20 utterances each and 1 label for each utterance -> 10x20=200 labels
# y_train = np.random.random_integers(0, 1, size=(100, ))
# y_train = to_categorical(y_train,num_classes=2)
# y_test = np.random.random_integers(0, 1, size=(100, ))
# y_test = to_categorical(y_test,num_classes=2)
# print "y_train ", y_train.shape
# vocab_size=100

rnn_model = model.RNN(x_train.shape[1], vocab_size, 2, embed_dim=32, emb_trainable=True, model_name="simple_rnn",
                      rnn_unit_type='rnn',loss_type="binary_crossentropy", hidden_dim=32, hidden_activation="relu",
                      out_activation="softmax",
                 bidirectional=False)
rnn_model.compile()
rnn_model.fit(x_train, y_train, validation_data=[x_test, y_test] , n_epochs=10,
              batch_size=100, callbacks=None, verbose=1)