from django.shortcuts import render
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def home(request):
    return render(request,"home.html")

def predict(request):
    return render(request,"predict.html")

def result(request):


    data = pd.read_csv(r'C:\Users\dpraj\Downloads\Diabetes detection using machine learning\diabetes.csv')

    le = LabelEncoder()
    for i in data.columns:
        data[i] = le.fit_transform(data[i])
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.21, random_state=6)

    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(Xtrain, Ytrain)

    l=[]
    val1 = float(request.GET['n1'])
    val1=val1-24
    val2 = str(request.GET['n2'])
    if val2.lower()=="male":
        val2=1
    elif val2.lower()=="female":
        val2=0
    val3 = str(request.GET['n3'])
    if val3.lower()=="yes":
        val3=1
    elif val3.lower()=="no":
        val3=0
    val4 = str(request.GET['n4'])
    if val4.lower()=="yes":
        val4=1
    elif val4.lower()=="no":
        val4=0
    val5 = str(request.GET['n5'])
    if val5.lower()=="yes":
        val5=1
    elif val5.lower()=="no":
        val5=0
    val6 = str(request.GET['n6'])
    if val6.lower()=="yes":
        val6=1
    elif val6.lower()=="no":
        val6=0
    val7 = str(request.GET['n7'])
    if val7.lower()=="yes":
        val7=1
    elif val7.lower()=="no":
        val7=0
    val8 = str(request.GET['n8'])
    if val8.lower()=="yes":
        val8=1
    elif val8.lower()=="no":
        val8=0
    val9 = str(request.GET['n9'])
    if val9.lower()=="yes":
        val9=1
    elif val9.lower()=="no":
        val9=0
    val10 = str(request.GET['n10'])
    if val10.lower()=="yes":
        val10=1
    elif val10.lower()=="no":
        val10=0
    val11 = str(request.GET['n11'])
    if val11.lower()=="yes":
        val11=1
    elif val11.lower()=="no":
        val11=0
    val12 = str(request.GET['n12'])
    if val12.lower()=="yes":
        val12=1
    elif val12.lower()=="no":
        val12=0
    val13 = str(request.GET['n13'])
    if val13.lower()=="yes":
        val13=1
    elif val13.lower()=="no":
        val13=0
    val14 = str(request.GET['n14'])
    if val14.lower()=="yes":
        val14=1
    elif val14.lower()=="no":
        val14=0
    val15 = str(request.GET['n15'])
    if val15.lower()=="yes":
        val15=1
    elif val15.lower()=="no":
        val15=0
    val16 = str(request.GET['n16'])
    if val16.lower()=="yes":
        val16=1
    elif val16.lower()=="no":
        val16=0


    pred = rfc.predict([[val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14,val15,val16]])
    prob = rfc.predict_proba([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16]])
    pos_prob = prob[pred == 1, 1]
    neg_prob = prob[:, 0]
    for i in pos_prob:
        pos = i * 100
    for j in neg_prob:
        neg = j * 100

    result1=""
    if pred==[1]:
        result1="You are {}% diabetic. Please visit nearest Healthcare Professional.".format(pos)
    else:
        result1="You are {}% Non diabetic.".format(neg)



    print(result)
    return render(request,"predict.html",{"result2":result1})