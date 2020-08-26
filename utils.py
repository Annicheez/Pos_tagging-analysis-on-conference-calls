import nltk
from nltk.corpus import stopwords, wordnet
from nltk.corpus import sentiwordnet as swn
import pandas as pd
import re


def tokenise(data: pd.Series):
    """This function takes a list of strings as input, tokenises it , and returns a list of
    tokens
    :param data:
    :return: tokenised_data
    """
    tokenised_data = []
    for row in data:
        new_row = nltk.word_tokenize(row)
        tokenised_data.append(new_row)
    return pd.Series(tokenised_data)


def remove_stopwords(data:pd.Series):
    """This function takes a pandas series object as input where each entry is a list of tokens
    and removes all stopwords
    :param data:
    :return: cleaned_data
    """
    cleaned_data = []
    stop_words = set(stopwords.words('english'))
    for row in data:
        filtered_sentence = [word for word in row if not word in stop_words and word.isalnum()]
        cleaned_data.append(filtered_sentence)
    return pd.Series(cleaned_data)


def lemmatize(data:pd.Series):
    """This function take a pandas series as input and lemmatizes each token
    entry in the series. The Wordnet lemmatizer is utilized.
    :param data:
    :return: lemmatized_data
    """
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_data = []
    for row in data:
        lemmatized_sentence = [lemmatizer.lemmatize(token) for token in row]
        lemmatized_data.append(lemmatized_sentence)
    return pd.Series(lemmatized_data)


def filter_pronouns(data:pd.Series):
    """This function takes a pandas series as input and filters for pronouns
    :param data:
    :return: filtered_data
    """
    filtered_data = []
    for row in data:
        filtered_tokens = [token for token in row if len(token) > 3]
        filtered_data.append(filtered_tokens)
    return pd.Series(filtered_data)


def filter_alphanumerics(data:pd.Series):
    """This function takes a pandas series as input and filters for alpha
    :param data:
    :return: filtered_data
    """
    filtered_data = []
    for row in data:
        filtered_tokens = [token for token in row if not bool(re.search('[0-9]+', token))]
        filtered_data.append((filtered_tokens))
    return pd.Series(filtered_data)

def pos_tagger(data:pd.Series):
    """"This function takes a pandas series as input and tags each word as per its
    part of speech classification
    :param data:
    :return: tagged_data
    """
    tagged_data =[]
    for row in data:
        tagged_sentence = nltk.pos_tag(row)
        tagged_data.append(tagged_sentence)
    return pd.Series(tagged_data)


def bigram_generate(data:pd.Series):
    """This function takes a pandas series as input, collapses it into a list of
    tokens and generates bigrams
    :param data:
    :return: bigrams
    """
    collapsed_data = [token for row in data for token in row]
    bigrams = nltk.bigrams(collapsed_data)
    return bigrams


def analyse_sentiment(data:pd.Series):
    """This function takes a pandas series as input, and tags each sentence according
    to its sentiment using sentinet
    :param data:
    :return: pd.Series
    """
    collapsed_data = [tupp for row in data for tupp in row]
    sentiment_words = []

    def penn_to_wn(tag):
        """
        Convert between the PennTreebank tags to simple Wordnet tags
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        elif tag.startswith('V'):
            return wordnet.VERB
        return None

    for word, pos in collapsed_data:
        wn_tag = penn_to_wn(pos)
        synsets = wordnet.synsets(word, pos = wn_tag)
        try:
            synset = synsets[0]
        except: pass
        swn_synset = swn.senti_synset(synset.name())
        tupp = (word, pos, swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score())
        sentiment_words.append(tupp)
    return sentiment_words

