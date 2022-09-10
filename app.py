from flask import Flask , render_template ,request

import tran as t
app = Flask(__name__,template_folder="Template")

@app.route('/',methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        return render_template('index.html')


@app.route('/sub', methods=['POST'])
def submit():
#   #html -> .py
    if request.method == 'POST':
        name = request.form.get('username')
        trans = t.prediction(name)
        return render_template("index.html",name = trans)
    return 'pl'

if __name__== "__main__":
    app.run(debug=True)