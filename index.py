from flask import Flask
import single_image_object_counting

app = Flask(__name__)

@app.route('/')
def hello_world():
    return single_image_object_counting.recognizeImage()