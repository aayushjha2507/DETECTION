from flask import Flask, render_template, request
import numpy as np
from joblib import load
app = Flask(__name__)

# rf = pickle.load(open("model.pkl", "rb"))
# vectorizer = pickle.load(open("vect.pkl", "rb"))

rf=load("compressmodel.joblib")
vectorizer=load("vectorizer_compressed.joblib")
@app.route("/", methods=["GET", "POST"])
def predict_news():
    if request.method == "POST":
        news_text = request.form['news']

        text_vect = vectorizer.transform([news_text])

        prdct = rf.predict(text_vect)

        if rf.predict_proba(text_vect)[0][0] < 0.00 or rf.predict_proba(text_vect)[0][0] == 0.0:
            result = "Fake news"
        else:
            result = "True news"

        return render_template("index.html", prediction=result, user_input=news_text)
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)