import os
import eventlet
import single_image_object_counting,calculate_item
from PIL import Image
from base64 import b64decode, b64encode
from io import BytesIO
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO


UPLOAD_FOLDER = './uploads'
AllOWED_EXTENSIONS = {'jpeg','jpg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on('process-image')
def process_image(sid,b64_image):
    raw_image = BytesIO()
    image_data = BytesIO(b64decode(b64_image[22:]))
    Image.open(image_data).convert('LA').save(raw_image, format='PNG')
    socketio.emit("processed-image", b64encode(raw_image.getvalue()).decode())

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

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
            foodDetected = single_image_object_counting.recognizeImage(str(imgUrl))
            print("CHECKTYPE")
            print(type(foodDetected))
            totalPrice = calculate_item.calculatePrice(foodDetected)
            imgUrl = url_for('send_file', filename=filename)
            return render_template('detection.html', imgUrl=imgUrl, foodDetected=foodDetected, totalPrice=totalPrice)
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