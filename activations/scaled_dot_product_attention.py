import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask = None):

    # tf.debugging.assert_equal(tf.shape(q)[0], tf.shape(k)[0])
    # tf.debugging.assert_equal(tf.shape(q)[0], tf.shape(v)[0])
    # tf.debugging.assert_equal(tf.shape(k)[1], tf.shape(v)[1])

    matmul_qk = tf.matmul(q, k, transpose_b=True)  

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  

    output = tf.matmul(attention_weights, v)  
    return output, attention_weights