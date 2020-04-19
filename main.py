import __main__
from TextProcessor import TextProcessor, Predition
from flask import Flask, render_template, request
import joblib


def dummy_fun(doc):
    return doc


application = Flask(__name__)
# dummy_fun = joblib.load('./models/model_tfidf_out.sav')
#

__main__.dummy_fun = dummy_fun
model_tfidf_out_name = './models/model_tfidf_out.sav'
prediction = Predition()
# tfidf_out = joblib.load(model_tfidf_out_name)
@application.route('/', methods=['POST', 'GET'])
def main():

    text = None
    result = None
    try:
        print(request.form)
        if 'text' in request.form:
            text = request.form['text']
            print('analis')
            ans = prediction.predict(text)
            result = 'fake' if ans == 1 else 'non fake'
            # Ваш текст здесь
            # можете вызвать просто здесь вызвать вашу функцию питона,
            # которая будет всё делать с параметром text
            # your_function(text)
        return render_template('index.html', text=result)
    except KeyError:
        return "KeyError"
