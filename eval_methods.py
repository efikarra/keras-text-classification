import model
import utils


def evaluate_ffn_model():
    loss=None
    return loss


def evaluate_rnn_model(ckpt_weights_file, x_data, y_data, batch_size, max_seq_length, vocab_size, n_classes, embed_dim,
                       emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
                       out_activation, bidirectional, learning_rate, verbose):
    rnn_model = model.RNN(max_seq_length=max_seq_length, vocab_size=vocab_size, n_classes=n_classes,
                          embed_dim=embed_dim, emb_trainable=emb_trainable, model_name=model_name,
                          rnn_unit_type=rnn_unit_type, loss_type=loss_type, hidden_dim=hidden_dim,
                          hidden_activation=hidden_activation, out_activation=out_activation,
                          bidirectional=bidirectional)
    utils.load_model(ckpt_weights_file, rnn_model, learning_rate)
    print("Model from checkpoint %s was loaded."%ckpt_weights_file)
    metrics_names, scores = rnn_model.evaluate(x_data, y_data, batch_size=batch_size, verbose=verbose)
    loss = scores[0]
    return loss


def evaluate(model_type, ckpt_weights_file, eval_res_folder, x_data_path, y_data_path, batch_size, max_seq_length, vocab_size, n_classes,
             embed_dim,
             emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
             out_activation, bidirectional, learning_rate, verbose):
    x_data, y_data = utils.create_model_input_data(x_data_path, y_data_path, max_seq_length)
    if model_type == "rnn":
        if y_data is not None:
            loss = evaluate_rnn_model(ckpt_weights_file, x_data, y_data, batch_size, max_seq_length, vocab_size, n_classes,
                               embed_dim,
                               emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
                               out_activation, bidirectional, learning_rate, verbose)
        predictions = predict_rnn_model(ckpt_weights_file, x_data, batch_size, max_seq_length, vocab_size, n_classes, embed_dim,
                          emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
                          out_activation, bidirectional, learning_rate, verbose)
        utils.save_predictions(eval_res_folder, predictions)
    elif model_type == "fnn":
        if y_data is not None:
            loss = evaluate_ffn_model()
        predictions=predict_ffn_model()
        utils.save(predictions)
    else:
        raise ValueError("Unknown model type. The supported types are: rnn|ffn")
    print("Loss: %.3f" % loss)


def predict_ffn_model():
    predictions=None
    return predictions


def predict_rnn_model(ckpt_weights_file, x_data, batch_size, max_seq_length, vocab_size, n_classes, embed_dim,
                      emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
                      out_activation, bidirectional, learning_rate, verbose):
    rnn_model = model.RNN(max_seq_length=max_seq_length, vocab_size=vocab_size, n_classes=n_classes,
                          embed_dim=embed_dim, emb_trainable=emb_trainable, model_name=model_name,
                          rnn_unit_type=rnn_unit_type, loss_type=loss_type, hidden_dim=hidden_dim,
                          hidden_activation=hidden_activation, out_activation=out_activation,
                          bidirectional=bidirectional)
    utils.load_model(ckpt_weights_file, rnn_model, learning_rate)
    predictions = rnn_model.predict(x_data, batch_size=batch_size, verbose=verbose)
    return predictions


def predict(model_type, ckpt_weights_file, x_data_path, batch_size, max_seq_length, vocab_size, n_classes, embed_dim,
            emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
            out_activation, bidirectional, learning_rate, verbose):
    x_data, y_data = utils.create_model_input_data(x_data_path, None, max_seq_length)
    if model_type == "rnn":
        predict_rnn_model(ckpt_weights_file, x_data, batch_size, max_seq_length, vocab_size, n_classes, embed_dim,
                          emb_trainable, model_name, rnn_unit_type, loss_type, hidden_dim, hidden_activation,
                          out_activation, bidirectional, learning_rate, verbose)
    elif model_type == "fnn":
        evaluate_ffn_model()
    else:
        raise ValueError("Unknown model type. The supported types are: rnn|ffn")
