# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:26:46 2022

@author: root
"""

#Import main library
import numpy as np

#Import Flask modules
from flask import Flask, request, render_template

#Import pickle to save our regression model
import pickle 

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'template')

#Open our model 
model = pickle.load(open('model.pkl','rb'))

#create our "home" route using the "index.html" page
@app.route('/home')
def home():
    return render_template('home.html')

#Set a post method to yield predictions on page
@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    
    #obtain all form values and place them in an array, convert into floats
    try:
        args=[]
        args.append(request.args.get('age'))
        args.append(request.args.get('weight'))
        args.append(request.args.get('height'))
        args.append(request.args.get('neck'))
        args.append(request.args.get('chest'))
        args.append(request.args.get('abd'))
        args.append(request.args.get('hip'))
        args.append(request.args.get('thigh'))
        args.append(request.args.get('knee'))
        args.append(request.args.get('ankle'))
        args.append(request.args.get('bic'))
        args.append(request.args.get('for'))
        args.append(request.args.get('wrist'))
        
        init_features = [float(x) for x in args]
        #Combine them all into a final numpy array
        final_features = [np.array(init_features)]
        #predict the bodyfat given the values inputted by user
        prediction = model.predict(final_features)
        
        #Round the output to 2 decimal places
        output = round(prediction[0],2)
        
       
        return render_template('output.html', prediction_text = 'Процент жира в Вашем организме равен примерно {}%'.format(output))   
    except:
        return 'Error' 

#Run app
if __name__ == "__main__":
    app.run(debug=True)