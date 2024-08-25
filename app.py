import numpy as np
import pandas as pd
import pickle
from flask import Flask,render_template,request

app=Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')
pipe=pickle.load(open("RidgeModel.pkl","rb"))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    locations=sorted(data['location'].unique())
    if request.method=="POST":
        location=request.form.get('location')
        sqft=float(request.form.get('sqft'))
        bath=float(request.form.get('bath'))
        bhk=float(request.form.get('bhk'))

        input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
        prediction=pipe.predict(input)[0] * 100000
        # prediction=pipe.predict(input)[0]
        prediction=np.round(prediction,2)
        detail="Predicted Price: ₹"
        
        if prediction<0:
            return "<p>Your Requirement doesn't fit well..........<br>The entered values seem unrealistic. Please check your inputs and try again.</p>" 
    
        return render_template('index.html',detail=detail,price=prediction,locations=locations)
    
    else:
        return "Can't Proceed! It may be due to any inappropriate inputs....."
    
if __name__=="__main__":
    app.run(debug=True)