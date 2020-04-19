from TextProcessor import TextProcessor, Predition
from flask import Flask, render_template, request
import joblib


def dummy_fun(doc):
    return doc


application = Flask(__name__)

prediction = Predition()


@application.route('/', methods=['POST', 'GET'])
def main():

    text = None
    result = None
    try:
        print(request.form)
        if 'text' in request.form:
            text = request.form['text']
            ans = prediction.predict(text)
            result = 'fake' if ans == 1 else 'non fake'
            # Ваш текст здесь
            # можете вызвать просто здесь вызвать вашу функцию питона,
            # которая будет всё делать с параметром text
            # your_function(text)
        return render_template('index.html', text=result)
    except KeyError:
        return "KeyError"
