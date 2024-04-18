import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix 
from svm import SVM
from sklearn.model_selection import train_test_split
from kernels import RBF


df = pd.read_csv('breast.csv')


df['Class'] = df['Class'].replace({'benign': 0, 'malignant': 1})
df['Bare_Nuclei'].replace('?', np.nan, inplace=True)
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')
df.dropna(subset=['Bare_Nuclei'], inplace=True)


print(df.keys())
      
df_feat = df.iloc[:, :-1]  # All rows, all columns except the last one
df_target = df.iloc[:, -1:]  # All rows, only the last column


# Target
df_target.columns = ['Class']
print(df_feat.dtypes)


# Print the first few rows of the features and target to verify
print(df_feat.head())
print(df_target.head())


print("Feature Variables: ") 
print(df_feat.info()) 


print("Dataframe looks like : ") 
print(df_feat.head()) 


X_train, X_test, y_train, y_test = train_test_split( 
						df_feat, np.ravel(df_target), 
				test_size = 0.30, random_state = 101) 

# train the model on train set 
#model = SVM.fit(X_train, y_train) 
model = SVM()
model.fit(X_train, y_train)

'''# print prediction results 
predictions = model.predict(X_test) 
print(classification_report(y_test, predictions))'''


predictions = model.predict(X_test)
predictions = np.where(predictions == -1, 0, predictions)  

print(classification_report(y_test, predictions))

C_values = [0.1, 1, 10, 100, 1000]
gamma_values = [1, 0.1, 0.01, 0.001, 0.0001]
best_score = 0

