from flask import Flask, render_template
import single_image_object_counting

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("./index.html")
    #return single_image_object_counting.recognizeImage()