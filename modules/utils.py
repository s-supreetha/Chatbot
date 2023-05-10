#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import streamlit as st

from modules.chatbot import Chatbot
from modules.embedder import Embedder
from langchain.callbacks import get_openai_callback



class Utilities:
    
    type=""
    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or from the user's input
        and returns it
        """
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
        else:
            user_api_key = st.sidebar.text_input(
                label="#### Your OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
            )
            if user_api_key:
                st.sidebar.success("API key loaded", icon="🚀")
        return user_api_key

    @staticmethod
    def handle_upload():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type=["csv","pdf","docx","txt"], label_visibility="collapsed")
        if uploaded_file is not None:
            

            def show_user_file(uploaded_file):
                filename=uploaded_file.name
                file_container = st.expander("Your file :")
                
                if filename.endswith(".csv"):
                    type="csv"
                    shows = pd.read_csv(uploaded_file,delimiter=',')
                    uploaded_file.seek(0)
                    file_container.write(shows)
                elif filename.endswith(".pdf"):
                    st.write("Pdf File")
                    type="pdf"
                elif filename.endswith(".pdf"):
                    st.write("Docx file")
                    type="docx"
                    
                elif filename.endswith(".txt"):
                    st.write("Text file")
                    type="txt"


            show_user_file(uploaded_file)
        else:
            st.sidebar.info(
                "👆 Upload your file to get started, "
              
            )
            st.session_state["reset_chat"] = True
        return uploaded_file

    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder()
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)
            chatbot = Chatbot(model, temperature, vectors)
        st.session_state["ready"] = True
        return chatbot

    def count_tokens_agent(agent, query):
        with get_openai_callback() as cb:
            result = agent(query)
            st.write(f'Spent a total of {cb.total_tokens} tokens')

        return result
    

