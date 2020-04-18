from flask import Flask, render_template, request


application = Flask(__name__)


@application.route('/', methods=['POST', 'GET'])
def main():
  text = None
  try:
    print(request.form)
    if 'text' in request.form:
      text = request.form['text']
      # Ваш текст здесь
      # можете вызвать просто здесь вызвать вашу функцию питона,
      # которая будет всё делать с параметром text
      # your_function(text)
    return render_template('index.html', text=text)
  except KeyError:
    return "KeyError"
  