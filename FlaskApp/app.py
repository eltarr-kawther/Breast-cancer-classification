from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np

from loadcnn import get_cnn
from loadxception import get_xception
 
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['GET','POST'])
def upload_image():
    M = "TL"
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        
        image_url = url_for('display_image', filename=filename)
        
        # preprocess image
        img = load_img("static/uploads/"+filename)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img_datagen = ImageDataGenerator(rescale=1/255.0)
        imgen = img_datagen.flow(img, batch_size=1)
        
        # load models
        if M == "TL":
            model = get_xception()
            model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=["accuracy"])
            model.build(input_shape=(None, 51, 51, 3))
            model.load_weights(r'C:\Users\straw\Desktop\AIS2\Breast-cancer-classification\data\models_weight\Xception_TLearning_weights.h5')

        else:
            model = get_cnn()
            model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=["accuracy"])
            model.build(input_shape=(None, 50, 50, 3))
            model.load_weights(r'C:\Users\straw\Desktop\AIS2\Breast-cancer-classification\data\models_weight\CNN_classic_weights.h5')
                
        # predict image
        y_pred = model.predict(imgen).argmax()
        return '''<h1>The prediction is: {}</h1> <img src="{}">'''.format(y_pred, image_url)
    
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(debug=True)