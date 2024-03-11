from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
import pandas as pd
import numpy as np
import seaborn as seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def home(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render())


def dashboard(request):
    template = loader.get_template('dashboard.html')
    return HttpResponse(template.render())


def prediction(request):
    # Getting data from diabetes data PIMA Indian Diabetes data set from github
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

    # Creating a dataframe
    df = pd.read_csv(url)

    # Heatmap shows that skin thickness and outcome have the lowest correlation almost close to 0
    # Data cleansing, remove the skin thickness variable by dropping the column
    df = df.drop(["SkinThickness"], axis=1)
    # df.info()

    # x contains independent variables and y is the dependent variable outcome which is the diagnostic prediction
    x = df.drop(["Outcome"], axis=1)
    y = df["Outcome"]

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(x.to_numpy(), y, test_size=0.3)

    # Train the model
    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)

    pregnancies = request.GET['pregnancies']
    age = request.GET['age']
    pedigree = request.GET['pedigree']
    bmi = request.GET['bmi']
    glucose = request.GET['glucose']
    blood_pressure = request.GET['blood_pressure']
    serum_insulin = request.GET['serum_insulin']

    inputs = [int(pregnancies), int(glucose), int(blood_pressure), int(serum_insulin), float(bmi), float(pedigree),
              int(age)]

    print(inputs)

    features = np.array([inputs])
    result = nb_model.predict(features)

    final_result = ""

    if result[0] == 1:
        final_result = "HIGH RISK"
    else:
        final_result = "LOW-AVERAGE RISK"

    # template = loader.get_template('prediction.html')
    # return HttpResponse(template.render())

    return render(request, "prediction.html", {"report": final_result})
