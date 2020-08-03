import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from nltk.corpus import stopwords
from keras.preprocessing import sequence

raw_data = pd.read_csv('bbc-text.csv')
df = raw_data.copy()

stop_words = set(stopwords.words('english'))
def preprocess_stop_words(text_list):
    res_list=[]
    for word in text_list:
        if word not in stop_words:
            res_list.append(word)

    return ' '.join(res_list)


df['pro_text'] = df['text'].str.split().apply(preprocess_stop_words)
X = df['pro_text']
y = df['category']

max_words = 5000
max_length = 250

tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(df['pro_text'].values)
X_seq = tok.texts_to_sequences(df['pro_text'].values)

X_seq = sequence.pad_sequences(X_seq, maxlen=max_length)
print('Shape of data tensor:', X_seq.shape)

y_seq = pd.get_dummies(df['category']).values

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Model Building

model = Sequential()

model.add(Embedding(max_words,64, input_length=max_length))
model.add(LSTM(50))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=20)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

fig, ax = plt.subplots(1,2,figsize=(15,5))
ax[0].plot(history.history['val_loss'], label='val_data' )
ax[0].plot(history.history['loss'], label='train_data')
ax[0].axis([1,12,0,2])
ax[0].set_title('Loss')
ax[0].legend()

ax[1].plot(history.history['val_accuracy'], label='val_data')
ax[1].plot(history.history['accuracy'], label='train_data')
ax[1].axis([1,12,0,2])
ax[1].set_title('Accuracy')
ax[1].legend()
plt.show()

model.save('Model_LSTM.h5py')
