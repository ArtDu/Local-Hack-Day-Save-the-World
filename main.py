from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def hello_world():
  text = None
  try:
    print(request.form)
    if 'text' in request.form:
      text = request.form['text']
    return render_template('index.html', text=text)
  except KeyError:
    return "KeyError"
  