import random
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/logo.png")
def logo():
    return render_template("logo.png")

@app.route("/logo-favicon.ico")
def favicon():
    return render_template("logo-favicon.ico")

def preprocess(input_list):
    total_list = []

    #  0   Patient Age                                      
    total_list.append(input_list['age'])
    #  1   Genes in mother's side                          
    total_list.append(1 if random.random() > 0.5 else 0)
    #  2   Inherited from father 
    total_list.append(1 if random.random() > 0.5 else 0)
    #  3   Maternal gene
    total_list.append(1 if random.random() > 0.5 else 0)
    #  4   Paternal gene  
    total_list.append(1 if random.random() > 0.5 else 0)
    #  5   Blood cell count (mcL)     
    total_list.append(input_list['bloodCellCount'])
    #  6   Mother's age  
    total_list.append(input_list['motherAge'])
    #  7   Father's age   
    total_list.append(input_list['motherAge'])  
    #  8   Status  
    total_list.append(1)    
    #  9   Respiratory Rate (breaths/min) 
    total_list.append(input_list['respiratoryRate'])
    #  10  Heart Rate (rates/min
    total_list.append(input_list['heartRate'])                             
    #  11  Test 1  
    total_list.append(0)                                           
    #  12  Test 2  
    total_list.append(0)                                             
    #  13  Test 3  
    total_list.append(0)                                             
    #  14  Test 4  
    total_list.append(1)                                             
    #  15  Test 5  
    total_list.append(0)                                             
    #  16  Parental consent 
    total_list.append(1)                                  
    #  17  Follow-up  
    total_list.append(1 if random.random() > 0.5 else 0)                                       
    #  18  Gender      
    total_list.append(input_list['gender'])                                      
    #  19  Birth asphyxia           
    total_list.append(input_list['birthAsphyxia'])                            
    #  20  Autopsy shows birth defect (if applicable)   
    total_list.append(random.choice([0, 1, 2, 3]))     
    #  21  Folic acid details (peri-conceptional)
    total_list.append(input_list['folicAcidDetails'])            
    #  22  H/O serious maternal illness
    total_list.append(input_list['hoMaternalIllness'])                      
    #  23  H/O radiation exposure (x-ray) 
    total_list.append(input_list['hoRadiationExposure'])                   
    #  24  H/O substance abuse            
    total_list.append(input_list['hoSubstanceAbuse'])                   
    #  25  Assisted conception IVF/ART 
    total_list.append(input_list['assistedConception'])                      
    #  26  History of anomalies in previous pregnancies   
    total_list.append(input_list['hoAnomalies'])   
    #  27  No. of previous abortion   
    total_list.append(input_list['prevAbortions'])                       
    #  28  Birth defects   
    total_list.append(input_list['birthDefects'])                                  
    #  29  White Blood cell count (thousand per microliter)  
    total_list.append(input_list['wbc'])
    #  30  Blood test result  
    total_list.append(input_list['bloodTestResult'])                               
    #  31  Symptom 1   
    total_list.append(0)                                       
    #  32  Symptom 2   
    total_list.append(1)                                       
    #  33  Symptom 3   
    total_list.append(1)                                       
    #  34  Symptom 4   
    total_list.append(1)                                       
    #  35  Symptom 5   
    total_list.append(0)                                       
    #  36  Genetic Disorder 
    total_list.append(random.choice([0, 1, 2]))                                  
    #  37  sum of Mother's and fathers age avg  
    motherAge = int(input_list['motherAge'][0])
    fatherAge = int(input_list['fatherAge'][0])
    average_age = (motherAge + fatherAge) / 2
    total_list.append(average_age)             
    #  38  total symptom                                     
    total_list.append(0.6)

    return np.array(total_list).reshape(1, 39) 

@app.route('/predict',methods=['POST'])
def predict():

    feature_list = request.form.to_dict()
    final_features = preprocess(feature_list)
    final_features = np.array(final_features, dtype=float)

    if(healthyCriteria(final_features) == True):
        finalPrediction = 9
    else:
        prediction = model.predict(final_features)
        finalPrediction = prediction[0]

    output = int(finalPrediction)
    if output == 0:
        text = "Cystic Fibrosis"
    elif output == 1:
        text = "Leber's Hereditary Optic Neuropathy"
    elif output == 2:
        text = "Diabetes"
    elif output == 3:
        text = "Leigh Syndrome"
    elif output == 4:
        text = "Cystic Fibrosis"
    elif output == 5:
        text = "Tay-Sachs"
    elif output == 6:
        text = "Hemochromatosis"
    elif output == 7:
        text = "Mitochondrial Myopathy"
    elif output == 8:
        text = "Alzheimer's Disease"
    else:
        text = "Healthy! 🙂"

    return render_template('verdict.html', prediction_text=text)

def healthyCriteria(input_list):
    if(input_list[0,9] == 1 and input_list[0,10] == 1 and  input_list[0,19] == 0):
        return True