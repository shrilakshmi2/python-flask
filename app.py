from flask import Flask,render_template,request
import os
from werkzeug.utils import secure_filename


app=Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction')
def predict():
    return render_template('dataloader.html')



@app.route('/uploadajax', methods = ['POST'])
def upldfile():
        
    prod_mas = request.files['prod_mas']
    filename = secure_filename(prod_mas.filename)
    prod_mas.save(os.path.join("./static/Upload/", filename))

    import pandas as pd
    dataset=pd.read_csv("./static/Upload/"+ filename)
    print(dataset)
    print('************************************')
    print(dataset.shape)
    print('************************************')
    print(dataset.head(5))
    print('************************************')
    print(dataset.info())
    print('************************************')
    from sklearn.model_selection import train_test_split
    predictors=dataset.drop('target',axis=1)#X Value
    target=dataset["target"]# Y Value

    #Dataset split 80:20
    X_train,X_test,Y_train,Y_test=train_test_split(predictors,target,test_size=0.20,random_state=0)

    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression

    lr=LogisticRegression()

    #fitness model
    lr.fit(X_train,Y_train)

    #Prediction
    y_pred_lr=lr.predict(X_test)

    #Accuracy Score
    print("Accuracy Score : "+str(accuracy_score(y_pred_lr,Y_test)*100))
        
    return render_template('dataloader.html',data="Data loaded successfully")


if __name__=="__main__":
    app.run(debug=True)
