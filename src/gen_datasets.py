# Imports
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import networkx as nx
from collections import OrderedDict
from itertools import combinations
import math
from tqdm import tqdm
import logging

# Constants
DATAFOLDER = "./data/"
FILTER_TOKENS = ". , ; & 's : ? ! ( ) ' ’ 'm 'no n't * *** -- ... / [ ]".split()
def dummy_func(x): return x


# Setup logging configuration
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


# Load a file from a pickle encoding
def load_pickle(filename):
    name = os.path.join(DATAFOLDER, filename)
    with open(name, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


# Save a file from a pickle encoding
def save_as_pickle(filename, data):
    name = os.path.join(DATAFOLDER, filename)
    with open(name, 'wb') as output:
        pickle.dump(data, output)


# Binomial coefficients gen
def nCr(n, r):
    f = math.factorial
    return int(f(n) / (f(r) * f(n - r)))


# Filter incomprehensible tokens and remove stopwords
def filter_tokens(tokens, stopwords):
    new_tokens = []
    for token in tokens:
        if (token not in stopwords) and (token not in FILTER_TOKENS):
            new_tokens.append(token)
    return new_tokens


def word_word_edges(p_ij):
    word_word = []
    cols = list(p_ij.columns)
    cols = [str(w) for w in cols]
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if p_ij.loc[w1, w2] > 0:
            word_word.append((w1, w2, {"weight": p_ij.loc[w1, w2]}))
    return word_word


def generate_text_graph(window=10):
    # generates graph based on text corpus
    # window = sliding window size to calculate point-wise mutual information between words
    # Is ten too large for the sliding window size? PMI should be less than that right?

    # Load in data from the tweets csv file
    logger.info("Preparing data...")
    df = pd.read_csv(os.path.join(DATAFOLDER, "tweets_combined.csv"))
    df.drop(["id"], axis=1, inplace=True)  # remove id field
    df = df[["tweet", "target"]]  # remove extraneous fields

    # Remove any stopwords to get simplified sentence structure
    stopwords = list(set(nltk.corpus.stopwords.words("english")))
    df["tweet"] = df["tweet"].apply(lambda x: nltk.word_tokenize(x.lower())).apply(lambda x: filter_tokens(x, stopwords))
    save_as_pickle("gen/df_data.pkl", df)  # parsed tweet data

    # Tfidf (term frequency - inverse document frequency)
    # In this case with only two classes, it looks for words that are strong indicators
    # These could clue in the nn that the tweet is strongly depressive or otherwise
    logger.info("Calculating Tf-idf...")
    vectorized = TfidfVectorizer(input="content", max_features=None, tokenizer=dummy_func, preprocessor=dummy_func)
    vectorized.fit(df["tweet"])
    df_tfidf = vectorized.transform(df["tweet"])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorized.get_feature_names_out()
    vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf, columns=vocab)

    # PMI (Pointwise mutual information) calculations
    # This establishes which words are likely to go together in the tweets
    names = vocab
    n_i = OrderedDict((name, 0) for name in names)
    word2index = OrderedDict((name, index) for index, name in enumerate(names))

    # Find the co-occurrences
    logger.info("Calculating co-occurrences...")
    no_windows = 0
    occurrences = np.zeros((len(names), len(names)), dtype=np.int32)
    for t in tqdm(df["tweet"], total=len(df["tweet"])):
        for i in range(len(t) - window):
            no_windows += 1
            d = set(t[i:(i + window)])

            for w in d:
                n_i[w] += 1
            for w1, w2 in combinations(d, 2):
                i1 = word2index[w1]
                i2 = word2index[w2]

                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1

    # The actual calculation bit
    logger.info("Calculating PMI...")
    p_ij = pd.DataFrame(occurrences, index=names, columns=names) / no_windows
    p_i = pd.Series(n_i, index=n_i.keys()) / no_windows

    # No longer need occurrence data
    del occurrences
    del n_i
    for col in p_ij.columns:
        p_ij[col] = p_ij[col] / p_i[col]
    for row in tqdm(p_ij.index, total=len(p_ij.index)):
        p_ij.loc[row, :] = p_ij.loc[row, :] / p_i[row]
    p_ij = p_ij + 1E-9
    for col in p_ij.columns:
        p_ij[col] = p_ij[col].apply(lambda x: math.log(x))

    # Build graph
    logger.info("Building graph (No. of document, word nodes: %d, %d)..." % (len(df_tfidf.index), len(vocab)))
    graph = nx.Graph()
    logger.info("Adding document nodes to graph...")
    graph.add_nodes_from(df_tfidf.index)  # Add the key words to the graph
    logger.info("Adding word nodes to graph...")
    graph.add_nodes_from(vocab)  # Other vocabulary added to the graph

    # Build edges between document-word pairs
    logger.info("Building document-word edges...")
    document_word = [(doc, w, {"weight": df_tfidf.loc[doc, w]}) for doc in
                     tqdm(df_tfidf.index, total=len(df_tfidf.index)) for w in df_tfidf.columns]

    logger.info("Building word-word edges...")
    word_word = word_word_edges(p_ij)
    save_as_pickle("gen/word_word_edges.pkl", word_word)
    logger.info("Adding document-word and word-word edges...")
    graph.add_edges_from(document_word)
    graph.add_edges_from(word_word)
    save_as_pickle("gen/text_graph.pkl", graph)
    logger.info("Done and saved!")


if __name__ == "__main__":
    generate_text_graph()
