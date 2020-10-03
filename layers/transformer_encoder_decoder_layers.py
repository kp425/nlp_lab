import tensorflow as tf
from tensorflow.keras import layers, Sequential
from .multihead_attention import MultiHeadAttention




def pointwise_ffn(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = pointwise_ffn(d_model, dff)

        self.norm_layer1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_layer2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, x, training, mask):

        attn_output,_ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm_layer1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm_layer2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = pointwise_ffn(d_model, dff)

        self.norm_layer1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_layer2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_layer3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        attn_output_1, attn_weights_1 = self.mha1(x, x, x, mask = look_ahead_mask)
        attn_output_1 = self.dropout1(attn_output_1, training=training)
        out1 = self.norm_layer1(attn_output_1+x)  # (batch_size, input_seq_len, d_model)

        attn_output_2, attn_weights_2 = self.mha2(out1, enc_output, enc_output, mask = padding_mask)
        attn_output_2 = self.dropout2(attn_output_2, training=training)
        out2 = self.norm_layer2(attn_output_2+out1)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.norm_layer3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
        return out3, attn_weights_1, attn_weights_2




