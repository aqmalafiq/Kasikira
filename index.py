import os
import eventlet
eventlet.monkey_patch()
import calculate_item
from PIL import Image
from base64 import b64decode, b64encode
from io import BytesIO
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from api.object_counting_api import mamakDetector
from utils import backbone
import numpy as np

UPLOAD_FOLDER = './uploads'
AllOWED_EXTENSIONS = {'jpeg','jpg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')
detection_graph, category_index = backbone.set_model('inference_graphGPU3', 'labelmap.pbtxt')
modal = mamakDetector(detection_graph) 

@socketio.on('process-image')
def process_image(b64_image,counter):
    image_data = BytesIO(b64decode(b64_image[22:]))
    modal.startSession(detection_graph)
    json,img = modal.detectStream(np.frombuffer(image_data.getvalue(), np.uint8),counter,category_index,0)
    socketio.emit("processed-image", {'image': 'b64encode(img).decode()','item': json})

@socketio.on('connect')
def test_connect():
    print('Client connected')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in AllOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template("index.html")
    

from flask import send_from_directory

@app.route('/show/<filename>', methods=['GET','POST'])
def uploaded_file(filename):
    if request.method == 'POST':
        if 'run' in request.form:
            imgUrl = request.form['run']
            modal.startSession(detection_graph)
            foodDetected = modal.detectSingleImage("."+str(imgUrl),detection_graph,category_index,0)
            modal.endSession()
            mydict = {}
            foodDetected = foodDetected.replace("'","")
            foodDetected = foodDetected.replace(" ","")
            splitted = foodDetected.split(",")

            for v in splitted:
                aux = v.split(":")
                mydict[aux[0]] = aux[2]
            
            imgUrl = url_for('send_file', filename=filename)
            return render_template('detection.html', imgUrl=imgUrl, foodDetected=mydict, totalPrice=calculate_item.calculatePrice(mydict))
    return render_template('upload.html', filename=filename)    

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/upload':  app.config['UPLOAD_FOLDER']
})

if __name__ == '__main__':
    socketio.run(app)