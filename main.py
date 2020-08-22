import psycopg2 as psy
import pandas as pd
import os
import dotenv
import nltk
from utils import *

dotenv.load_dotenv()

connection = psy.connect(host = os.environ['pghost'], port = 5432, database = os.environ['pgdatabase'],
                            user = os.environ['pgusername'], password = os.environ['pgpassword'])
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

data = pd.read_sql(sql_query, connection)
token_data  = tokenise(data.speaker_text)
ctoken_data = remove_stopwords(token_data)
tagged_data = pos_tagging(ctoken_data)

print(tagged_data[3])


