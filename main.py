from TextProcessor import TextProcessor, Predition
from flask import Flask, render_template, request
import joblib

application = Flask(__name__)

model_class_name = './models/model_classification.sav'
model_w2v_name = './models/model_w2v.sav'

model_w2v = joblib.load(model_w2v_name)
model_classification = joblib.load(model_class_name)

text_processor = TextProcessor(model_w2v)
model_prediction = Predition(model_classification)


@application.route('/', methods=['POST', 'GET'])
def main():
    text = None
    try:
        print(request.form)
        if 'text' in request.form:
            text = request.form['text']
            text_vec = text_processor.transform(text)
            result = model_prediction.predict(text_vec)
            # Ваш текст здесь
            # можете вызвать просто здесь вызвать вашу функцию питона,
            # которая будет всё делать с параметром text
            # your_function(text)
        return render_template('index.html', text=result)
    except KeyError:
        return "KeyError"
