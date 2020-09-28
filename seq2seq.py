import os
#must do before importing tensorflow
os.environ['PYTHONHASHSEED']=str(2)

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

import numpy as np
import random
import pandas as pd
import sys

import matplotlib.pyplot as plt
import seaborn as sns

import string

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(2)
   tf.random.set_seed(2)
   np.random.seed(2)
   random.seed(2)






class DataOps:
    def __init__(self, text):
        self.vocab = list(sorted(set(text)))
        self.vocab_size = len(self.vocab)
        self.vocab2id = {u:i for i,u in enumerate(self.vocab)}
    
    def encode(self, string_):
        return [self.vocab2id[i] for i in string_]
    
    def decode(self, list_of_ints):
        return ''.join([self.vocab[i] for i in list_of_ints])


sample_size = 700

lang1 = [''.join([random.choice(string.ascii_uppercase) for i in range(1,random.randint(2,11))]) for _ in range(sample_size)]
lang2 = [''.join([str(ord(i)) for i in j]) for j in lang1]


lang1_info = DataOps(''.join(lang1))
lang2_info = DataOps(''.join(lang2))

#teacher forcing
lang2_info.vocab.append("<start>")
lang2_info.vocab.append("<end>")
lang2_info.vocab2id["<start>"] = lang2_info.vocab_size
lang2_info.vocab2id["<end>"] = lang2_info.vocab_size + 1
lang2_info.vocab_size += 2

print(lang2_info.vocab2id)


padded_lang1 = tf.keras.preprocessing.sequence.pad_sequences(list(map(lang1_info.encode,lang1)),padding="post")
padded_lang2 = tf.keras.preprocessing.sequence.pad_sequences(list(map(lambda x:[lang2_info.vocab2id["<start>"]]+lang2_info.encode(x)+[lang2_info.vocab2id["<end>"]],lang2)),padding="post")


train_split = 550


train, test = (padded_lang1[:train_split], padded_lang2[:train_split]) ,\
                (padded_lang1[train_split:], padded_lang2[train_split:])


# print(len(train[0]), len(train[1]))
# print(len(test[0]), len(test[1]))

def inp_targ_split(lang1, lang2):
    lang1_inp = lang1
    lang2_inp = lang2[:-1]
    lang2_targ = lang2[1:]
    return lang1_inp, lang2_inp, lang2_targ

batch_size = 16

ds = tf.data.Dataset.from_tensor_slices(train)
ds = ds.map(inp_targ_split).shuffle(10000, reshuffle_each_iteration = True)
ds = ds.batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices(test).map(inp_targ_split)
test_batch_size = 100000
test_ds = test_ds.batch(test_batch_size)



enc_embedd_dims = 8
n_enc_tokens = lang1_info.vocab_size

dec_embedd_dims = 8
n_dec_tokens = lang2_info.vocab_size

lstm_units = 64



class Encoder(Model):
    def __init__(self, n_tokens, embedding_dims, units, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.embedding_layer = layers.Embedding(n_tokens, embedding_dims, mask_zero = True)
        self.lstm1 = layers.LSTM(units, return_sequences=True)
        self.lstm2 = layers.LSTM(units, return_state=True)

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.lstm1(x)
        _,enc_h, enc_c = self.lstm2(x)
        return enc_h, enc_c


class Decoder(Model):
    def __init__(self, n_tokens, embedding_dims, units, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.embedding_layer = layers.Embedding(n_tokens, embedding_dims, mask_zero=True)
        self.lstm1 = layers.LSTM(units, return_sequences=True)
        self.lstm2 = layers.LSTM(units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(n_tokens)

    def call(self, inputs, initial_states):
        x = self.embedding_layer(inputs)
        x = self.lstm1(x, initial_state = initial_states)
        x, h, c = self.lstm2(x)
        output = self.dense(x)
        return output, h, c

enc_model = Encoder(n_enc_tokens, enc_embedd_dims, lstm_units)
dec_model = Decoder(n_dec_tokens, dec_embedd_dims, lstm_units)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam()


def forward_pass(ds_chunk):
    enc_x, dec_x, dec_y = ds_chunk
    enc_h, enc_c = enc_model(enc_x)
    dec_preds,_,_ = dec_model(dec_x,[enc_h, enc_c])
    return dec_preds, dec_y
        

def train_step(ds_chunk):
    with tf.GradientTape() as tape:
        y_preds, y_true = forward_pass(ds_chunk)
        loss_val = loss_fn(y_true, y_preds)
    train_vars = enc_model.trainable_variables + dec_model.trainable_variables
    grads = tape.gradient(loss_val, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))
    return loss_val



def train(train_ds, val_ds, epochs= 60, save_every=100):
    losses, accuracies = [],[]
    loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for epoch in range(epochs):
        for count,chunk in enumerate(ds):
            loss_val = train_step(chunk)
            loss_avg.update_state(loss_val)
            y_preds, y_true = forward_pass(next(iter(val_ds)))
            accuracy.update_state(y_true, y_preds)
            sys.stdout.write(f"\rEpoch {epoch+1}({count+1}/{train_ds.cardinality()})        loss: {loss_avg.result():.3f}          acc: { accuracy.result():.3f}")
        print()
        losses.append(loss_avg.result().numpy())
        accuracies.append(accuracy.result().numpy())
        loss_avg.reset_states()
        accuracy.reset_states()

        # if (epoch+1) % save_every == 0:
        #     # pathlib.Path("folder").mkdir(parents=True, exist_ok=True)
        #     model.save(filepath+"model_epoch_{}".format(epoch))

    return (losses, accuracies)


history = train(train_ds = ds, val_ds = test_ds)