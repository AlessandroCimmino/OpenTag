import keras
from keras_self_attention import SeqSelfAttention
from keras_contrib.layers import CRF


def build_model(tag_num,
                max_seq_len,
                features=1024,
                embedding_dim=100,
                embedding_weights=None,
                rnn_units=100,
                return_attention=False,
                lr=1e-3):
    """Build the model for predicting tags.

    :param token_num: Number of tokens in the word dictionary.
    :param tag_num: Number of tags.
    :param embedding_dim: The output dimension of the embedding layer.
    :param embedding_weights: Initial weights for embedding layer.
    :param rnn_units: The number of RNN units in a single direction.
    :param return_attention: Whether to return the attention matrix.
    :param lr: Learning rate of optimizer.
    :return model: The built model.
    """

    input_layer = keras.layers.Input(shape=(max_seq_len,features))
    lstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=rnn_units,
                                                              recurrent_dropout=0.4,
                                                              return_sequences=True),
                                            name='Bi-LSTM')(input_layer)
    attention_layer = SeqSelfAttention(attention_activation='sigmoid',
                                attention_width=9,
                                return_attention=return_attention,
                                name='Attention')(lstm_layer)
    if return_attention:
        attention_layer, attention = attention_layer
    crf = CRF(tag_num, sparse_target=True, name='CRF')

    outputs = [crf(attention_layer)]
    loss = {'CRF': crf.loss_function}
    if return_attention:
        outputs.append(attention)
        loss['Attention'] = 'categorical_crossentropy'

    model = keras.models.Model(inputs=input_layer, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=loss,
        metrics={'CRF': crf.accuracy},
    )
    return model
