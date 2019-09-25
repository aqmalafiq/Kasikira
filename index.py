import os
import single_image_object_counting
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename



UPLOAD_FOLDER = './uploads'
AllOWED_EXTENSIONS = {'jpeg','jpg'}

app = Flask(__name__)
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
            imgUrl = url_for('send_file', filename=filename)
            return render_template('detection.html', imgUrl=imgUrl, foodDetected=foodDetected)
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

