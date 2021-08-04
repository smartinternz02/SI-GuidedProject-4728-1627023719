import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app=Flask(__name__)
global sess
sess = tf.Session()
global graph
graph=tf.compat.v1.get_default_graph()
set_session(sess)

model = load_model("vegetable.h5")
model1=load_model("fruitnames.h5")

@app.route('/')
def Home():
    return render_template("Home.html")

@app.route('/Predict')
def GetStarted():
    return render_template('Predict.html')

@app.route('/Predict',methods=['GET','POST'])		
def Predict():
    if request.method == 'POST':
        
        a = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(a.filename))
        a.save(file_path)
        img = image.load_img(file_path, target_size=(128, 128))
        
        b = image.img_to_array(img)
        b = np.expand_dims(b, axis=0)
        preds = model.predict(b)
        index = ["Pepper_bell___Bacterial_spot","Pepper_bell___healthy","Potato___Early_blight","Potato___Late_blight","Potato___healthy","Tomato___Bacterial_spot","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot"]
        index1 =["Apple___Black_rot","Apple___healthy","Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy","Peach___Bacterial_spot","Peach___healthy"]
        plant=request.form['plant']
        print(plant)
        if(plant=="vegetable"):
            with graph.as_default():
                set_session(sess)
                preds = model.predict_classes(b)
            print(preds)
            df=pd.read_excel('precautions - veg.xlsx')
            print(df.iloc[preds[0]]['caution'])
        else:
            with graph.as_default():
                set_session(sess)
                preds = model1.predict_classes(b)
                
            df=pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[preds[0]]['caution'])
            result=str(index[preds[0]])
        return result
    
@app.route('/Logout')
def Logout():
    return render_template('Logout.html')

if __name__ ==  "__main__":
     app.run(debug=True)

