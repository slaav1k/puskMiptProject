
import numpy as np

def softmax(vector):
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    weighted_decoder = np.dot(decoder_hidden_state.T, W_mult)
    attention_scores = np.dot(weighted_decoder, encoder_hidden_states)
    softmax_weights = softmax(attention_scores)
    attention_vector = np.dot(softmax_weights, encoder_hidden_states.T).T
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    enc_transformed = np.dot(W_add_enc, encoder_hidden_states)
    dec_transformed = np.dot(W_add_dec, decoder_hidden_state)
    combined = enc_transformed + dec_transformed
    tanh_combined = np.tanh(combined)
    attention_scores = np.dot(v_add.T, tanh_combined)
    softmax_weights = softmax(attention_scores)
    attention_vector = np.dot(softmax_weights, encoder_hidden_states.T).T
    return attention_vector
