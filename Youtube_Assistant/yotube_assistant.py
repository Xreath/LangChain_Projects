import Langchain_Helper.lc_helper as lch

import streamlit as st
import textwrap

st.title("YouTube Assistant")

with st.sidebar:

    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="What is the YouTube video URL?",
            max_chars=100
            )
        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=100,
            key="query"
            )
      
        submit_button = st.form_submit_button(label='Submit')


        
if query and youtube_url:

    db = lch.create_vectordb_from_youtube_url(youtube_url)
    response, docs = lch.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=85))