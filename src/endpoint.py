from predict import talk




from PIL import Image
from flask import Flask, request
import json
from waitress import serve
import base64
app = Flask(__name__)

from flask import render_template

cpu = "--"

@app.route("/")
def root():
    # Loads the index.html in templates/
    return render_template('index.html', message="Hola PR!")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    content = request.get_json(silent=True)
    print('REQUEST:',content)
    # Got image encoded in base 64, need to convert it to png

    if(content and content['recipe']):
        recipestring = content['recipe']
        print(recipestring)
    else:
        recipestring = "vodka ; beer ; sugar ; lime ; Stir all ingredients with ice and strain into a big cocktail glass . Add the sugar on the top and serve"

    result = "['John circus','death beach']"
    try:
        result = str(talk(recipestring))
    except Exception as e:
        print('ERROR: Printing default values',e)

    if(result):
        print("PREDICTION:",result)
        return json.dumps({'Status':'OK','prediction':result})
    else:
        return json.dumps({"Status":"ERROR"})

@app.route("/job")
def job():
    return json.dumps({'Status':'OK'})

if __name__ == "__main__":
    serve(app,host='0.0.0.0', port=5001)


