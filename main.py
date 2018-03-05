import train_methods
import eval_methods

vocab_size = 2000
n_classes = 2
embed_dim = 32
emb_trainable = True
model_name = "rnn_test"
rnn_unit_type = 'rnn'
loss_type = "binary_crossentropy"
hidden_dim = 32
hidden_activation = "relu"
out_activation = "sigmoid"
bidirectional = False
n_epochs = 20
batch_size = 100
learning_rate = 0.01
save_checkpoint = True
early_stopping = True
verbose = 1
max_seq_length = 100
x_train_path = "experiments/data/x_train_subset.pickle"
y_train_path = "experiments/data/y_train_subset.pickle"
x_val_path = "experiments/data/x_test_subset.pickle"
y_val_path = "experiments/data/y_test_subset.pickle"
model_type = "rnn"
eval_weights_ckpt = "experiments/trained_models/rnn_test.02-0.667.hdf5"
eval_x_data = "experiments/data/x_test_subset.pickle"
eval_y_data = "experiments/data/y_test_subset.pickle"
eval_res_folder = "experiments/results"

if __name__ == "__main__":

    if eval_weights_ckpt is not None:
        eval_methods.evaluate(model_type, eval_weights_ckpt, eval_res_folder,
                              eval_x_data, eval_y_data, batch_size, max_seq_length,
                              vocab_size, n_classes,
                              embed_dim,
                              emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
                              out_activation, bidirectional, learning_rate, verbose)
    else:
        train_methods.train(x_train_path=x_train_path, y_train_path=y_train_path,
                            x_val_path=x_val_path,
                            y_val_path=y_val_path, model_type=model_type,
                            max_seq_length=max_seq_length, vocab_size=vocab_size, n_classes=n_classes,
                            embed_dim=embed_dim,
                            emb_trainable=emb_trainable,
                            model_name=model_name, rnn_unit_type=rnn_unit_type, loss_type=loss_type,
                            hidden_dim=hidden_dim, hidden_activation=hidden_activation,
                            out_activation=out_activation,
                            bidirectional=bidirectional, n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate,
                            save_checkpoint=save_checkpoint, early_stopping=early_stopping, verbose=verbose
                            )
