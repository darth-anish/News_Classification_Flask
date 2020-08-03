import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from nltk.corpus import stopwords


df = pd.read_csv('ML/bbc-text.csv', engine='python')
stop_words = set(stopwords.words('english'))

def preprocess_stop_words(text_list):
    res_list=[]
    for word in text_list:
        if word not in stop_words:
            res_list.append(word)

    return ' '.join(res_list)


df['pro_text'] = df['text'].str.split().apply(preprocess_stop_words)

max_words = 5000
max_length = 250
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(df['pro_text'].values)
X_seq = tok.texts_to_sequences(df['pro_text'].values)
X_seq = sequence.pad_sequences(X_seq, maxlen=max_length)
