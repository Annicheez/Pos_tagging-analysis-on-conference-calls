import psycopg2 as psy
import pandas as pd
import os
import dotenv
import nltk
from utils import *
import matplotlib.pyplot as plt


def main():
    # Anonymously loading login credentials for the MCCGR database
    dotenv.load_dotenv()

    # Establishing connection
    connection = psy.connect(host = os.environ['pghost'], port = 5432, database = os.environ['pgdatabase'],
                                user = os.environ['pgusername'], password = os.environ['pgpassword'])

    # SQL query to load data
    sql_query = """SELECT speaker_data.file_name, speaker_data.last_update, call_files.file_name, call_files.mtime, 
    call_files.ctime, company_ids.cusip, speaker_data.speaker_name, speaker_data.role, speaker_data.context, 
    speaker_data.speaker_number,speaker_data.speaker_text
    FROM streetevents.speaker_data
    INNER JOIN streetevents.call_files
        ON speaker_data.file_name = call_files.file_name
    INNER JOIN streetevents.company_ids 
        ON speaker_data.file_name = company_ids.file_name
    LIMIT 100
    """
    # Reading the sql query reply into a pandas dataframe
    data = pd.read_sql(sql_query, connection)

    # Tokenize data
    token_data = tokenise(data.speaker_text)

    # Remove stopwords
    ctoken_data = remove_stopwords(token_data)

    # Lemmatize and filter for pronouns and alphanumerics
    cltoken_data = filter_alphanumerics(filter_pronouns(lemmatize(ctoken_data)))

    # POS tagging
    tagged_data = pos_tagger(cltoken_data)

    # Isolating adjectives
    # Collapsing tagged data into one list of tuples
    collapsed_tags = [tupp for row in tagged_data for tupp in row]

    # Isolating adjectives
    adjectives = [word for (word, tag) in collapsed_tags if tag in ('JJ', 'JJR', 'JJS')]

    # Summarising adjectives
    tags_fd = nltk.FreqDist(adjectives)

    # Understanding usage of adjectives
    # Converting data to bigrams
    trigrams = nltk.trigrams(collapsed_tags)
    predecessors = [a[1] for (a, b, c) in trigrams if c[1] in ('JJ', 'JS', 'JR')]
    nltk.FreqDist(predecessors)

    # Reinitialising the trigram generator object
    trigrams = nltk.trigrams(collapsed_tags)
    successors = [b[0] for (a, b, c) in trigrams if c[1] in ('JJ', 'JS', 'JR')]
    nltk.FreqDist(successors)

    print(nltk.FreqDist(word for (word, tag) in collapsed_tags).most_common(1000))
    sents = analyse_sentiment(tagged_data)
    print(sents)
    # bigrams = bigram_generate(tagged_data)
    # adj_predec = [a[1] for (a,b) in bigrams if b[1] in ('JJ', 'JJR', 'JJS')]
    # print(nltk.FreqDist(adj_predec).most_common())


if __name__ == '__main__':
    main()