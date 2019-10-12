from flask import Flask,render_template,url_for,request
#from flask_material import Material
#we import material so we can rap it around our app for to asses it
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd




#EDA PACKAGES
import pandas as pd
import numpy as np

# MACHINE PACKAGES
import pickle
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


iris = load_iris()

x= iris.data
y= iris.target
knn= KNeighborsClassifier()
knn.fit(x,y)

log = LogisticRegression()

x_train,x_test,y_train,y_test =train_test_split(x, y, test_size =0.1, random_state = 0)
log.fit(x, y)
# r u can import pickels but i did not even save any data so no stress


app= Flask(__name__, template_folder='template')
#
#Material(app)

@app.route('/')
def index():
	return render_template("index.html")

#@app.route('/')
#def index():
#	return render_template("preview.html")



#another route  for our predictions
@app.route('/analyze', methods=['POST'])
def analyze():
	if request.method =='POST':
		sepal_length = request.form['sepal_length']
		sepal_width = request.form['sepal_width']
		petal_length = request.form['petal_length']
		petal_width = request.form['petal_width']
		model_choice= request.form['model_choice']
		sample_data = [sepal_length, sepal_width, petal_length, petal_width]
		#change from unicode to float
		clean_data = [ float(i) for i in sample_data]
		#reshaping
		ex1 = np.array(clean_data).reshape(1, -1)
		#ex1 =np.array(ex1)
		#that is to take each samples singly and treat individually

		# now lets write the condition statements for the models
		if model_choice == 'logmodel':
			model = joblib.load("data/finalized_model.sav")
			result =model.predict(ex1)
		else:
			model_choice == 'knnmodel'
			model = joblib.load("data/finalize_model.sav")
			result =model.predict(ex1)
		if result == 0:
			black = ('setosa')
		elif result == 1:
			black = ('vesicolor')
		else:
			black = ('verginica')
	return render_template("index.html", sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width, clean_data=clean_data, model_choice=model_choice, result=result, black=black)


if __name__ == '__main__':
	app.run(debug=True)