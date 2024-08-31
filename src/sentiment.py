import nltk
import xlsxwriter
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

"""
    Processing sentiment analysis from the Isolation Journals
    and packaging for data visualization
"""

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
# Journal entries are not published on github
corpus_dir = '../data/'
corpus = PlaintextCorpusReader(corpus_dir, '.*')

words = [str.lower(w) for w in corpus.words() if w.isalpha()]
fd = nltk.FreqDist(words)
bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(words)
trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
quadgram_finder = nltk.collocations.QuadgramCollocationFinder.from_words(words)
words = np.unique(words).tolist()

#Create pandas dataframe for negative and positive words
df_dict = {}
columns = ["Word", "Score", "Frequency"]
pos_df = pd.DataFrame(columns=columns)
neg_df = pd.DataFrame(columns=columns)

for w in words:
    score = sia.polarity_scores(w)
    if score["neg"] > 0.5:
        neg_df = pd.concat([pd.DataFrame([[w, score["neg"], fd[w]]], columns=columns), neg_df],
                       ignore_index=True)
    if score["pos"] > 0.5:
        pos_df = pd.concat([pd.DataFrame([[w, score["pos"], fd[w]]], columns=columns), pos_df], 
                        ignore_index=True)

df_dict["Positive Words"] = pos_df
df_dict["Negative Words"] = neg_df

# Create dataframe of negative and positive bigrams
num = 200
ngram_names = ["Bigrams", "Trigrams", "Quadgrams"]
ngram_finders = [bigram_finder, trigram_finder, quadgram_finder]

for ind in range(len(ngram_names)):
    columns = [ngram_names[ind], "Pos Score", "Neg Score", "Frequency"]
    ngram_df = pd.DataFrame(columns=columns)
    ngram_list = ngram_finders[ind].ngram_fd.most_common(num)
    for ngram_tup in ngram_list:
        ngram = ngram_tup[0]
        ngram_str = " ".join(ngram)
        score = sia.polarity_scores(ngram_str)
        if (score["neg"] > 0.3) or (score["pos"] > 0.3) and ("suleika" not in ngram_str):
            ngram_df = pd.concat([pd.DataFrame([[ngram_str, score["pos"], score["neg"], ngram_tup[1]]],
                                columns=columns), ngram_df], ignore_index=True)
    df_dict[ngram_names[ind]] = ngram_df

# Create excel files
writer = ExcelWriter("isolation_journal_data.xlsx")
for key in df_dict.keys():
        df_dict[key].to_excel(writer, sheet_name=key, engine="xlsxwriter")
writer.close()