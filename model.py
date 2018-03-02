import os
import h5py

from keras.models import Model
from keras.layers import Dense, SimpleRNN, LSTM, Input, Embedding, Bidirectional


class BaseRNNModel(object):
    def __init__(self, n_classes, model_name="test", rnn_unit_type="rnn", loss_type="binary_crossentropy",
                 out_activation="softmax"):
        self.model_name = model_name
        self.rnn_unit_type = rnn_unit_type
        self.n_classes = n_classes
        self.model = None
        self.loss_type = loss_type
        self.out_activation=out_activation

    def compile(self, metrics=[], optimizer='adam'):
        self.model.compile(loss=self.loss_type, optimizer=optimizer, metrics=[self.loss_type] +metrics)

    def fit(self, x_train, y_train, validation_data=None , n_epochs=10, batch_size=100, callbacks=None, verbose=1):
        return self.model.fit(x_train, y_train, validation_data=validation_data, epochs=n_epochs,
                                 batch_size=batch_size, verbose=verbose, callbacks=callbacks)

    def predict(self, x_test, batch_size=10, verbose=1):
        return self.model.predict(x_test, batch_size=batch_size, verbose=verbose)

    def evaluate(self, x_test, y_test, batch_size=10, verbose=0):
        scores = self.model.evaluate(x_test, y_test, verbose=verbose, batch_size=batch_size)
        return self.model.metrics_names, scores

    def save_weights(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = h5py.File(folder + self.model_name + ".h5", 'w')
        weights = self.model.get_weights()
        for i in range(len(weights)):
            file.create_dataset('weight' + str(i), data=weights[i])
        file.close()

    def load_model_weights(self, filepath):
        # workaround to load weights into new model
        # f = h5py.File(filepath, 'r')
        # weight = []
        # for i in range(len(f.keys())):
        #     weight.append(f['weight' + str(i)][:])
        # self.model.set_weights(weight)
        self.model.load_weights(filepath, by_name=False)


class RNN(BaseRNNModel):

    def __init__(self, max_seq_length, features, n_classes, embed_dim=32, emb_trainable=True,
                 model_name="simple_rnn", rnn_unit_type='rnn',
                 loss_type="binary_crossentropy", hidden_dim=32, hidden_activation="relu", out_activation="softmax",
                 bidirectional=False):
        BaseRNNModel.__init__(self, n_classes, model_name, rnn_unit_type, loss_type, out_activation)
        self.out_activation = out_activation
        self.bidirectional = bidirectional
        self.max_seq_length = max_seq_length
        self.features = features
        self.embed_dim = embed_dim
        self.emb_trainable = emb_trainable
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.build_model()


    def build_model(self):
        # the model receives sequences of length self.max_seq_length. At each timestep, the vector size is self.features.
        input = Input(shape=(self.max_seq_length, self.features), dtype='int32', name="myinput")
        # the input sequence is encoded into dense vectors of size self.embed_dim.
        # the input value 0 is a special padding value (for sequences with variable length)
        # that should be masked out. Our vocabulary SHOULD start from 1.
        embedded_input = self.embedding_layer()(input)

        # A RNN or a LSTM transforms the input sequences into a single vector (which is the last hidden state of the rnn)
        # This vector has size self.hidden_dim.
        if self.rnn_unit_type == 'rnn':
            recurrent_layer = SimpleRNN(self.hidden_dim, activation=self.hidden_activation, name="rnn")
        elif self.rnn_unit_type == 'lstm':
            recurrent_layer = LSTM(self.hidden_dim, activation=self.hidden_activation, name="lstm")
        else:
            raise ValueError('Unknown model type')
        # For Bidirectional rnn, the forward and backward states will be concatenated. So the output vector
        # will have size self.hidden_dim*2.
        if self.bidirectional:
            recurrent_layer = Bidirectional(recurrent_layer, merge_mode="concat")(embedded_input)
        else: recurrent_layer = recurrent_layer(embedded_input)

        if self.loss_type=="binary_crossentropy" or self.loss_type=="sparse_categorical_crossentropy":
            out_dim=1
        elif self.loss_type=="categorical_crossentropy":
            out_dim=self.n_classes
        else: raise ValueError("The allowed loss functions are binary_crossentropy, categorical_crossentropy, "
                               "sparse_categorical_crossentropy.")

        # The output layer takes as input the last hidden state of the rnn and produces a probability distribution
        # over classes.
        preds = Dense(out_dim, activation=self.out_activation, name="output")(recurrent_layer)
        self.model = Model(inputs=input, outputs=preds)


    def embedding_layer(self):
        return Embedding(input_dim=self.features, output_dim=self.embed_dim, mask_zero=True,
                         input_length=self.max_seq_length,
                         trainable=self.emb_trainable, name="embedding")