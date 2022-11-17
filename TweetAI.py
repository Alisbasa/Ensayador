import numpy as np
from keras.models import load_model
import streamlit as st
from flask import Flask, request, jsonify, render_template, url_for
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer

def main():
    
    tokenizer = GPT2Tokenizer.from_pretrained('./model')
    model = TFGPT2LMHeadModel.from_pretrained('./model')
    st.title("Bienvenido a tweetAI")
    st.subheader("Escribe una frase y genera un Tweet al estilo AMLO")
    value=st.text_input("Escribe aquí la oración inicial")
    if st.button("Generar Tweet"):
        input_ids = tokenizer.encode(value, return_tensors='tf')
        beam_output = model.generate(
            input_ids,
            max_length = 24,
            num_beams = 5,
            temperature = 0.7,
            no_repeat_ngram_size=2,
            num_return_sequences=5
        )
        print("un gato")
        st.success(tokenizer.decode(beam_output[0]))

if __name__ == '__main__':
    main()
    
