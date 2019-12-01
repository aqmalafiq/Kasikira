from flask import Flask,render_template
from io import BytesIO
from flask_socketio import SocketIO
from api.object_counting_api import mamakDetector
from base64 import b64decode,b64encode
from utils import backbone
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)
detection_graph, category_index = backbone.set_model('inference_graphGPU3','labelmap.pbtxt')
modal = mamakDetector(detection_graph)

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on("process-image")
def process_image(b64_image):
    print("Server Received Image")
    # print(b64_image)
    image_data = BytesIO(b64decode(b64_image[22:]))
    modal.startSession(detection_graph)
    json,img = modal.detectStream(np.frombuffer(image_data.getvalue(), np.uint8),0,category_index,0)
    socketio.emit("processed-image", b64encode(img).decode())
    mydict = {}
    json = json.replace("'","")
    json = json.replace(" ","")
    splitted = json.split(",")

    for v in splitted:
        aux = v.split(":")
        if len(aux) > 1:
            mydict[aux[0]] = aux[2] 
    print(mydict)
    socketio.emit("processed-text",mydict)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload')
def uploadHome():
    return render_template("upload.html")

if __name__ == '__main__':
    socketio.run(app)