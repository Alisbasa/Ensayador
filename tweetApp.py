# -*- coding: utf-8 -*-

import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
import time
import re

#"""Esta sección es parte de lo que se verá en la web"""

st.write("""
# Generador de Tweets
Escribe como comenzará tu tweet y nuestra APP redactará el resto al estilo de AMLO!
""")
contexto = 'hola'
contexto = st.text_input('Contexto:', 'Escribe el inicio del tweet')


#"""Leemos el dataset y le quitamos columnas que no necesitamos"""
st.write("""
### Este es el ultimo tweet que tomamos en cuenta: 
""")

tweets = pd.read_csv('tweets.csv', index_col = False)
#tweets.head()
tweets.drop('id', inplace=True, axis=1)
tweets.drop('created_at', inplace=True, axis=1)
tweets.drop('favorite_count', inplace=True, axis=1)
tweets.drop('retweet_count', inplace=True, axis=1)
#tweets.head()

tweets.size

#tweets.head(1)['text'].values[0]

#"""Removing links on the tweets"""

import re
for i in range(len(tweets)):
  tweets.loc[i,"text"] = re.sub(r' https://t.co/\w{10}', '', tweets.loc[i,"text"])

tweets.head(1)['text'].values[0]

#"""Vectorizando los datos"""

with open('tweets.txt', "w", encoding="utf-8") as my_output_file:
    for  row in tweets.head(500).itertuples():
      [my_output_file.write("".join(str(row[1]))+'\n') ]
#my_output_file.close()

text = open('tweets.txt', 'rb').read().decode(encoding='utf-8' ,errors='ignore')
vocab = sorted(set(text))
#print(vocab)

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
#all_ids

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

for ids in ids_dataset.take(10):
    chars_from_ids(ids).numpy().decode('utf-8')

seq_length = 100
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
#   print(chars_from_ids(seq))

   def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# for seq in sequences.take(5):
#   print(text_from_ids(seq).numpy())

#"""Función que regresa un dataset de pares (input, label), donde estos pares el input es una letra de una cadena, y el label la letra que sigue."""

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

#split_input_target(list("Tensorflow"))

dataset = sequences.map(split_input_target)

# for input_example, target_example in dataset.take(1):
#     print("Input :", text_from_ids(input_example).numpy())
#     print("Target:", text_from_ids(target_example).numpy())

#"""Creando batches"""

# Batch size
BATCH_SIZE = 64

# Tamaño de búfer para mezclar el conjunto de datos
# (Los datos TF están diseñados para trabajar con secuencias posiblemente infinitas,
# para que no intente barajar toda la secuencia en la memoria. En cambio,
# mantiene un búfer en el que mezcla elementos).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

model =tf.saved_model.load(
    './memoria'
)

#"""## Probamos el modelo
#Para ver si funciona correctamente hacemos pluebas con las muestras que aislamos anteriormente

#"""





# """# Generando Texto"""

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Creamos una mascara para prevenir que se generen carateres "[UNK]".
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # ponemos -inf en cada index erroneo.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # hacemos que coincida la forma con el vocabulario
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
   # return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)



start = time.time()
states = None
next_char = tf.constant([contexto])
result = [next_char]

for n in range(240):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
result2 = result[0].numpy().decode('utf-8'), '\n\n' + '_'*80
print(result2)
print('\nRun time:', end - start)
st.write(result2)