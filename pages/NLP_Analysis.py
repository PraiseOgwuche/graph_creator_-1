import streamlit as st
import pandas as pd
from backend.text_analysis import TextAnalyser
from io import StringIO
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Text Analysis")
st.write("This tool will help you to analyze text")
st.sidebar.header("Text Analysis Parameters")
uploaded_file = st.sidebar.file_uploader("Upload dataframe", type='csv')
if uploaded_file:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()
    dataframe = pd.read_csv(uploaded_file)
    text_analyzer = TextAnalyser(dataframe)
    dataframe.columns = dataframe.columns.str.strip()
    with st.sidebar:
        column = st.selectbox('Select column for analysis', options=dataframe.columns)

        analysis_type = st.selectbox("Select what you want to do with the data",
                                     options=["Wordcloud", "Sentiment Analysis"])

    if analysis_type == "Wordcloud":
        num_of_words = st.sidebar.number_input("Maximum number of words in WordCloud",
                                               min_value=1, max_value=200, value=20)
        st.subheader("The WordCloud generated for your data:")
        wordcloud = text_analyzer.draw_word_cloud(column, num_of_words=num_of_words)
        plt.imshow(wordcloud) # image show
        plt.axis('off') # to off the axis of x and y
        st.pyplot()

    elif analysis_type == "Sentiment Analysis":
        text_analyzer.sentiment_analysis(column)

