import pandas as pd
import string
from wordcloud import WordCloud, STOPWORDS


class TextAnalyser:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def draw_word_cloud(self, column: str, num_of_words: int):
        df = self.df.loc[1:, :].reset_index(drop=True)
        data = df[column]
        s = " ".join([response.lower() for response in data if type(response) == str]
                     ).translate(str.maketrans('', '', string.punctuation))
        wordcloud = WordCloud(stopwords=STOPWORDS, collocations=True,
                              background_color='white',
                              width=1500,
                              height=700, max_words=num_of_words
                              ).generate(s)
        return wordcloud

    def sentiment_analysis(self, column: int):
        pass
