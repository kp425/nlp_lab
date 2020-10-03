import tensorflow as tf
from tensorflow.keras import layers
from nlp_lab.activations import scaled_dot_product_attention

class MultiHeadAttention(layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0
        self.depth_of_each_head = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth_of_each_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, keys, values, mask = None):
        
        batch_size = tf.shape(query)[0]

        query = self.wq(query)
        keys = self.wk(keys)
        values = self.wv(values)

        query = self.split_into_heads(query, batch_size)
        keys = self.split_into_heads(keys, batch_size)
        values = self.split_into_heads(values, batch_size)

        attention, attention_weights = scaled_dot_product_attention(query, keys, values, mask)
        # scores = tf.matmul(query, keys, transpose_b=True)
        # scaled_scores = (1/(self.depth_of_each_head**0.5)) * scores
        # attention_weights = tf.nn.softmax(scaled_scores, axis = -1)
        # attention = tf.matmul(attention_weights, values)
        concat_attention = tf.reshape(attention, 
                                  (batch_size, -1, self.d_model)) 
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


