import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import time

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


##Import all Classifiers

from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier

df = pd.read_csv('./data/heart.csv')

def data_enhancement(df):
    
    df_gen = df
    


    age_std = df_gen['age'].std()
    trtbps_std = df_gen['trtbps'].std()
    chol_std = df_gen['chol'].std()
    thalachh_std = df_gen['thalachh'].std()
    oldpeak_std = df_gen['oldpeak'].std()    
    
    for i in df_gen.index:
        if np.random.randint(2) == 1:
            df_gen['age'].values[i] += age_std/7
        else:
            df_gen['age'].values[i] -= age_std/7
              
        if np.random.randint(2) == 1:
            
            df_gen['trtbps'].values[i] += trtbps_std/5
            
        else:
            df_gen['trtbps'].values[i] -= trtbps_std/5
                
        if np.random.randint(2) == 1:
            df_gen['chol'].values[i] += chol_std/10
        else:
            df_gen['chol'].values[i] -= chol_std/10
        
        if np.random.randint(2) == 1:
            df_gen['thalachh'].values[i] += thalachh_std/5
        else:
            df_gen['thalachh'].values[i] -= thalachh_std/5
   
        if np.random.randint(2) == 1:
            df_gen['oldpeak'].values[i] += oldpeak_std/10
        else:
            df_gen['oldpeak'].values[i] -= oldpeak_std/10

            
    return df_gen


def data_visualisation(df):

	x = df.drop('output',axis = 1)

	#Countplot

	plt.figure(figsize=(15, 8))
	sns.countplot(x=df["age"]);  # using countplot
	plt.title("Count of samples Vs Age", fontsize=20)
	plt.xlabel("Age", fontsize=20)
	plt.ylabel("Samples", fontsize=20)
	plt.show()

	#Correlation Matrix
	plt.figure(figsize=(15, 10))
	matrix = np.triu(x.corr())
	sns.heatmap(x.corr(), annot=True, linewidth=.8, mask=matrix, cmap="PuBuGn")
	plt.title("Correlation Matrix without target field", fontsize=20)
	plt.show()

	fig, axes = plt.subplots(2, 4, sharex=True, figsize=(10,10))
	fig.suptitle('Target events by different parameters')
	sns.countplot(x="sex", data=df, hue="output",ax=axes[0,0])
	sns.countplot(x="cp", data=df, hue="output",ax=axes[0,1])
	sns.countplot(x="fbs", data=df, hue="output",ax=axes[0,2])
	sns.countplot(x="restecg", data=df, hue="output",ax=axes[0,3])
	sns.countplot(x="exng", data=df, hue="output",ax=axes[1,0])
	sns.countplot(x="slp", data=df, hue="output",ax=axes[1,1])
	sns.countplot(x="caa", data=df, hue="output",ax=axes[1,2])
	sns.countplot(x="thall", data=df, hue="output",ax=axes[1,3])
	plt.show()

	fig, axes = plt.subplots(2, 3, sharex=True, figsize=(10,10))
	fig.suptitle('Boxplots of the Numeral Fields ')
	axes[0,0].set_title('Target Vs Age')
	axes[0,1].set_title('Target Vs Ttbps')
	axes[0,2].set_title('Target Vs Chol')
	axes[1,0].set_title('Target Vs Talachh')
	axes[1,1].set_title('Target Vs oldpeak')
    
	sns.boxplot(x="output", y="age" , data=df,  orient='v' , ax=axes[0,0])
	sns.boxplot(x="output", y="trtbps", data=df,  orient='v' , ax=axes[0,1])
	sns.boxplot(x="output", y="chol", data=df,orient='v', ax=axes[0,2])
	sns.boxplot(x="output", y="thalachh", data=df,orient='v',ax=axes[1,0])
	sns.boxplot(x="output", y="caa", data=df,orient='v',ax=axes[1,1])
	plt.show()
 
	return df


def model_analisys(df):
	
## Preprocessing

  X, y = df.drop('output', axis = 1 ), df['output']

  cat_var = ['sex', 'cp', 'fbs','restecg', 'exng', 'slp','caa','thall' ]
  num_var = ['age','trtbps','chol','thalachh', 'oldpeak' ]

  pn = Pipeline ( [ ('scaling', StandardScaler() ) ] )

  pc = Pipeline( [ ('ord',OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)) ]) 


  prep = ColumnTransformer(transformers=[('cat',pc, cat_var),('num', pn, num_var)]) 


## Models definition 

  tree_classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Random Forest":RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Skl GBM": GradientBoostingClassifier(),
    "Skl HistGBM": HistGradientBoostingClassifier() ,
    "XGBoost": XGBClassifier(use_label_encoder=False) ,
    "LightGBM":LGBMClassifier() ,
    "CatBoost": CatBoostClassifier(verbose = False)
      }


  d1 = {}

  for n,v in tree_classifiers.items():
      #print(n,v)
      p = Pipeline([('prep', prep),(n,v) ]) 
      d1.update({n:p})
      #tree_classifiers.update({n:p})
      #print(k)

  tree_classifiers = {name: make_pipeline(prep, model) for name, model in tree_classifiers.items()}


  ## Models beauty contest

  skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

  results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

  for model_name, model in tree_classifiers.items():
     
    
    start_time = time.time()

    pred = cross_val_predict(model, X, y,cv=skf)

    total_time = time.time() - start_time
    
    results = results.append({"Model":    model_name,
                              "Accuracy": accuracy_score(y, pred)*100,
                              "Bal Acc.": balanced_accuracy_score(y, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
                          


  results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
  results_ord.index += 1 
  
  print(results_ord)
  
  results_ord.to_csv('results.csv')
  return(results_ord)



def run_model(df, df_gen):
    
	cat_var = ['sex', 'cp', 'fbs','restecg', 'exng', 'slp','caa','thall' ]
	num_var = ['age','trbps','chol','thalachh', 'oldpeak' ]

	pn = Pipeline ( [ ('scaling', StandardScaler() ) ] )
	pc = Pipeline( [ ('ord',OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)) ]) 

	prep = ColumnTransformer(transformers=[('cat',pc, cat_var),('num', pn, num_var)]) 

	X, y = df.drop('output', axis = 1 ), df['output']
   
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0,stratify = y)

	enh_sample = df_gen.sample(df_gen.shape[0] // 3)
    
	X_train = pd.concat([X_train, enh_sample.drop(['output'], axis=1 ) ])
	y_train = pd.concat([y_train, enh_sample['output'] ])

	# Select the model according with the results from model_selection

	model =   RandomForestClassifier()

	tree_classifiers = {
		"Decision Tree": DecisionTreeClassifier(),
		"Extra Trees": ExtraTreesClassifier(),
		"Random Forest":RandomForestClassifier(),
		"AdaBoost": AdaBoostClassifier(),
		"Skl GBM": GradientBoostingClassifier(),
		"Skl HistGBM": HistGradientBoostingClassifier() ,
		"XGBoost": XGBClassifier(use_label_encoder=False) ,
		"LightGBM":LGBMClassifier() ,
		"CatBoost": CatBoostClassifier(verbose = False)
	}
	model.fit(X_train,y_train)
    
	y_pred =  model.predict(X_test)
    
	acc = accuracy_score(y_test, y_pred)
	#print(acc)
	print('The confusion matrix is a follows:')
	print(confusion_matrix(y_test, y_pred))
    
	save = input('Do you want to save the model? Y/N:  ')
	if save in ('Y', 'y'):
		model_name = './model/optimal_model.pkl'
		with open(model_name, 'wb') as file:
			pickle.dump(model, file)
			print("The model has been saved in ./model/optimal_model")
	return

def predict_patient():

	model = pickle.load(open('./model/optimal_model.pkl','rb'))


	print(type(model), model)


	while True:
		print('Welcome to the Heart Attack Predictor.')

		age = int(input('How old is the patient? (Integer) \n'))
		sex = int(input('Gender (0 Male, 1 Female) \n'))
		cp = int(input('Chest Pain type type? (0-3)\n'))
		trtbps = int(input('Resting blood pressure (mm HG) \n'))
		chol = int(input('Cholesterol (mg/dl) \n'))
		fbs  = int(input('Fasting blodd suger higher than 120 (0 = No, 1, Yes) \n'))
		restecg = int(input('Resting electrocardiographic results \n'))
		thalachh = int(input('Maximum heart rate achieved. \n'))
		exng = int(input('Exercise induce angina (0 No, 1 Yes) \n'))
		oldpeak = float(input('Previous peak \n'))
		slp = int(input('SLP \n'))
		caa  = int(input('NUmber of major vessels (0,4) \n'))
		thall  = int(input('Thall \n'))
    
		#lst =     [age, sex , cp, trtbps, chol, fbs, restcg, thalachh, exng, oldpeak, slp, caa,thall ]
		#col = ['age', 'sex', 'cp','trtbps' , 'chol', 'fbs','restcg','restcg','thalachh','exng','oldpeak', 'slp', 'caa','thall'] 
    
		d1 = {
			'age' : [int(age)],
			'sex' : [int(sex)],
			'cp' : [int(cp)],
			'trtbps' : [int(trtbps)],
			'chol' : [int(chol)],
			'fbs'  : [int(fbs)],
			'restecg' : [int(restecg)],
			'thalachh' : [int(thalachh)],
			'exng' : [int(exng)],
			'oldpeak' : [int(oldpeak)],
			'slp' : [int(slp)],
			'caa'  : [int(caa)],
			'thall'  : [int(thall)]
    		}
    
    
		df1 = pd.DataFrame.from_dict(d1) 
		print(df1)

		input('Press Enter to continue...')


		result = model.predict(df1)
    
		if result == 1:
			print ('\nThe patient is in Danger, according to out data.\n')
		else:
			print('\nThe patient does not fit in the pattern for the drug.\n')
        
		print('\n\n')
		time.sleep(4)
		input("Press Enter to continue...")
		print('\n\n')
	return

def main(df):
    
 
	while True:
		print('\n\nWelcome to the Heart Attack Predictor Model Cheker.')
		print('\n Please select an option.')
		print('\n\n 1. To see the original data.')
		print('\n 2. To analyse the models.')
		print('\n 3. To train the model with the best classifier found. ')
		print('\n 4. Predict Values.\n\n')

		res = (input('Which option do you want? \t'))
   

		if res == '1':
			data_visualisation(df)
  
		elif res == '2':
			
			model_analisys(df)

		elif res == '3':
			df_gen = data_enhancement(df)
			run_model(df, df_gen) 

		elif res == '4':
			predict_patient()

	return
        
if __name__ == '__main__':
	main(df)