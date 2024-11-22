from flask import Flask,render_template,request
import pickle
from sklearn.feature_extraction.text import CountVectorizer

total = 0
ham = 0
spam = 0

app = Flask(__name__)

@app.route("/")
@app.route("/home")

def home():
    return render_template("index.html")


@app.route("/result",methods=["POST","GET"])
def result():

    global total
    global ham
    global spam
    
    # vectorizer = CountVectorizer()

    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    with open('spam.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    output = request.form.to_dict()

    l = [output["Name"]]

    prediction = loaded_model.predict(vectorizer.transform(l))

    total = total + 1

    if(prediction[0]=="ham"):
        ham = ham + 1
    else:
        spam = spam + 1

    return render_template("index.html",p=prediction[0],t=total,h=ham,s=spam)

    # if(prediction[0]==ham):
    #     prediction[0]="This is not a spam message :)"
    #     return render_template("index.html",p=prediction[0])
    # else:
    #     prediction[0]="This is a spam message :("
    #     return render_template("index.html",p=prediction[0])

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
