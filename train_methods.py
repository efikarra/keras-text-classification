from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad
import model
import os
import utils


def train_model(model, x_train, y_train, validation_data, n_epochs, batch_size, learning_rate,
                early_stopping=True, save_checkpoint=True, verbose=1, out_dir="trained_models"):
    callbacks = []
    if save_checkpoint:
        # save the model at every epoch. 'val_loss' is the monitored quantity.
        # If save_best_only=True, the model with the best monitored quantity is not overwitten.
        # If save_weights_only=True, only the model weights are saved calling the method model.save_weights
        checkpoint = ModelCheckpoint(os.path.join(out_dir,model.model_name + ".{epoch:02d}-{val_loss:.3f}.hdf5"),
                                          verbose=verbose, monitor='val_loss', save_weights_only=True, save_best_only=True)
        callbacks.append(checkpoint)
    if early_stopping:
        # Training stops when the monitored quantity (val_loss) stops improving.
        # patience is the number of epochs with no improvement after which training is stopped.
        stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=15, verbose=verbose, mode='auto')
        callbacks.append(stopping)
    adam = Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0, clipnorm=1.)
    model.compile(metrics=[], optimizer=adam)
    history = model.fit(x_train, y_train, validation_data=validation_data, n_epochs=n_epochs,
                        batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    return history


def train_ffn_model():
    pass


def train_rnn_model(x_train, y_train, x_val, y_val, max_seq_length, vocab_size, n_classes, embed_dim,
                    emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
                    out_activation, bidirectional, n_epochs, batch_size, learning_rate, save_checkpoint,
                    early_stopping, verbose):
    rnn_model = model.RNN(max_seq_length=max_seq_length, vocab_size=vocab_size, n_classes=n_classes, embed_dim=embed_dim,
                          emb_trainable=emb_trainable, model_name=model_name, rnn_unit_type=rnn_unit_type,
                          loss_type=loss_type, hidden_dim=hidden_dim, hidden_activation=hidden_activation,
                          out_activation=out_activation, bidirectional=bidirectional)
    history = train_model(rnn_model, x_train, y_train, validation_data=(x_val, y_val), save_checkpoint=save_checkpoint,
                          n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                          early_stopping=early_stopping, learning_rate=learning_rate)
    return utils.extract_results_from_history(history)


def train(x_train_path, y_train_path, x_val_path, y_val_path,model_type,
                        max_seq_length, vocab_size, n_classes,
                        embed_dim,
                        emb_trainable,
                        model_name, rnn_unit_type, loss_type,
                        hidden_dim, hidden_activation,
                        out_activation,
                        bidirectional, n_epochs, batch_size,
                        learning_rate,
                        save_checkpoint, early_stopping, verbose):
    x_train, y_train = utils.create_model_input_data(x_train_path, y_train_path, max_seq_length)
    x_val, x_val = utils.create_model_input_data(x_val_path, y_val_path, max_seq_length)
    if model_type=="rnn":
        train_rnn_model(x_train, y_train, x_val, x_val, max_seq_length, vocab_size, n_classes, embed_dim,
                    emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
                    out_activation, bidirectional, n_epochs, batch_size, learning_rate, save_checkpoint,
                    early_stopping, verbose)
    elif model_type=="fnn":
        train_ffn_model()
    else: raise ValueError("Unknown model type. The supported types are: rnn|ffn")