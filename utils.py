import nltk
from nltk.corpus import stopwords
import pandas as pd

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
    :return:cleaned_column
    """
    cleaned_column = []
    stop_words = set(stopwords.words('english'))
    for row in data:
        filtered_sentence = [word for word in row if not word in stop_words and word.isalnum()]
        cleaned_column.append(filtered_sentence)
    return pd.Series(cleaned_column)


def pos_tagging(data:pd.Series):
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
