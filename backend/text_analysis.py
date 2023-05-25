import pandas as pd
import string
from wordcloud import WordCloud, STOPWORDS
import textnets as tn
import spacy
from typing import List
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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

    def sentiment_analysis(self, column: int, method: str, sensitivity: float):
        data = [r.lower() for r in self.df[column] if type(r) == str]
        if method == "ASENT":
            return self.asent_method(data, sensitivity)
        elif method == "TextBlob":
            return self.textblob_method(data, sensitivity)
        elif method == "VaderSentiment":
            return self.vadersentiment_method(data, sensitivity)

    def asent_method(self, data: List[str], sensitivity: float):
        # load spacy pipeline
        nlp = spacy.blank('en')
        nlp.add_pipe('sentencizer')

        # add the rule-based sentiment model
        nlp.add_pipe('asent_en_v1')

        positives = []
        negatives = []
        ans = set()
        for answer in data:
            doc = nlp(answer)
            if answer in ans:
                continue
            else:
                ans.add(answer)
            if doc._.polarity.positive > sensitivity:
                positives.append((answer, doc, doc._.polarity.positive))
            elif doc._.polarity.negative > sensitivity:
                negatives.append((answer, doc, doc._.polarity.negative))
        positives.sort(key=lambda i: i[2])
        negatives.sort(key=lambda i: i[2])
        return positives, negatives, len(data)

    def textblob_method(self, data: List[str], sensitivity: float):
        positives = []
        negatives = []
        ans = set()
        higher_bound = 1 - sensitivity
        lower_bound = sensitivity
        for answer in data:
            testimonial = TextBlob(answer)
            if answer in ans:
                continue
            else:
                ans.add(answer)
            if testimonial.sentiment.polarity > higher_bound:
                positives.append((answer, testimonial.sentiment.polarity - 0.5))
            elif testimonial.sentiment.polarity < lower_bound:
                negatives.append((answer, 0.5 - testimonial.sentiment.polarity))
        positives.sort(key=lambda i: i[1])
        negatives.sort(key=lambda i: i[1])
        return positives, negatives, len(data)

    def vadersentiment_method(self, data: List[str], sensitivity: float):
        positives = []
        negatives = []
        ans = set()
        for answer in data:
            analyzer = SentimentIntensityAnalyzer()
            vs = analyzer.polarity_scores(answer)
            if answer in ans:
                continue
            else:
                ans.add(answer)
            if vs['pos'] > sensitivity:
                positives.append((answer, vs['pos']))
            elif vs['neg'] > sensitivity:
                negatives.append((answer, vs['neg']))
        positives.sort(key=lambda i: i[1])
        negatives.sort(key=lambda i: i[1])
        return positives, negatives, len(data)

    def text_network_analysis(self, column: int, group_column: int):
        df = self.df[[column, group_column]]
        df = df.set_index(group_column)
        corpus = tn.Corpus(pd.Series(df[column], dtype="string"))
        t = tn.Textnet(corpus.tokenized(), min_docs=1)
        return t.plot(label_nodes=True, show_clusters=True)

