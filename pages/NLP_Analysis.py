import streamlit as st
import pandas as pd
from backend.text_analysis import TextAnalyser
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
import asent
import streamlit.components.v1 as components

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
                                     options=["Wordcloud", "Sentiment Analysis", "Text Network Analysis"])

    if analysis_type == "Wordcloud":
        num_of_words = st.sidebar.number_input("Maximum number of words in WordCloud",
                                               min_value=1, max_value=200, value=20)
        st.subheader("The WordCloud generated for your data:")
        wordcloud = text_analyzer.draw_word_cloud(column, num_of_words=num_of_words)
        plt.imshow(wordcloud)  # image show
        plt.axis('off')  # to off the axis of x and y
        st.pyplot()

    elif analysis_type == "Sentiment Analysis":
        method = st.sidebar.selectbox("Select the algorithm to perform Sentiment Analysis",
                                      options=["ASENT", "TextBlob", "VaderSentiment"])
        if method == "TextBlob":
            sensitivity = st.sidebar.number_input(label="Input sensitivity to positive and negative "
                                                        "(smaller sensitivity, more sentiment)",
                                                  min_value=0.01, max_value=0.99, value=0.2)
        else:
            sensitivity = st.sidebar.number_input(label="Input sensitivity to positive and negative "
                                                        "(smaller sensitivity, more sentiment)",
                                                  min_value=0.01, max_value=0.5, value=0.2)
        positives, negatives, len_ans = text_analyzer.sentiment_analysis(column, method, sensitivity)
        st.subheader('Statistics:')
        st.write(f'Percentage of positive responses: {round((len(positives) / len_ans) * 100, 2)}%')
        st.write(f'Percentage of negative responses: {round((len(negatives) / len_ans) * 100, 2)}%')
        st.subheader(f'Top {len(positives[-5:])} positive responses:')
        for ind, positive in enumerate(positives[-5:]):
            if method == 'ASENT':
                components.html(asent.visualize(positive[1], style="prediction"), height=100, scrolling=True)
            else:
                st.write(ind + 1, positive[0])
        st.subheader(f'Top {len(negatives[-5:])} negative responses:')
        for ind, negative in enumerate(negatives[-5:]):
            if method == 'ASENT':
                components.html(asent.visualize(negative[1], style="prediction"), height=100, scrolling=True)
            else:
                st.write(ind + 1, negative[0])

    elif analysis_type == "Text Network Analysis":
        group_column = st.sidebar.selectbox('Select group column for analysis', options=dataframe.columns)
        plot = text_analyzer.text_network_analysis(column, group_column)
        plot.save('plot.png')
        img = Image.open('plot.png')
        st.image(img)
