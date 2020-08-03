from flask import Flask, render_template, request
from ML.nlp_preprocess import tok,preprocess_stop_words,max_length
from tensorflow.keras.preprocessing import sequence
import numpy as np
from tensorflow.keras.models import load_model


app = Flask(__name__, template_folder='templates')

#model loading
model = load_model('ML/Model_LSTM.h5py')

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        var = request.form.get('Feed')

    labels=['Business','Entertainment','Politics','Sports','Tech']

    def predict_res(sample):
        sam_list = sample.split()
        test = preprocess_stop_words(sam_list)
        testing = tok.texts_to_sequences([test])
        sam_seq = sequence.pad_sequences(testing, maxlen=max_length)
        res = model.predict(sam_seq)
        return labels[np.argmax(res)]

    pred_class = predict_res(var)

    return render_template('predict.html', result = pred_class)



app.run(debug=True)
