import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pick
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np



def create_model(data):
  X=data[['cgpa','sgpa', '10th_percentage', '12th_percentage','certification_platform','social_media',
       'clubs_joined','avg_time_on_social_media']]
  y=data['Fast_slow_learner']
  
  #scale the data
  scaler=StandardScaler()
  X=scaler.fit_transform(X)
  
  #split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.35,random_state=42
  )
  
  #train the model
  # Create an instance of the SimpleImputer class
  imputer = SimpleImputer(strategy='median')

  # Fit and transform the imputer on the training data
  X_train_imputed = imputer.fit_transform(X_train)

  # Transform the imputer on the test data
  X_test_imputed = imputer.transform(X_test)

  # Create a new XGBoost classifier
  xgb_model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, random_state=42)

  # Fit the model on the imputed training data
  #xgb_model.fit(X_train_imputed, y_train)
  weights = np.ones(X_train_imputed.shape[0])  # Initialize weights for each sample
  xgb_model.fit(X_train_imputed, y_train, sample_weight=weights)

  # Predict on the imputed test data
  y_pred = xgb_model.predict(X_test_imputed)

  # Calculate accuracy score
  accuracy_XGB = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy_XGB)
  print("Classification report: \n", classification_report(y_test, y_pred))
  
  return xgb_model,scaler



  
def normalize(data,col_name):
  # Get the maximum value in the column
  max_val = data[col_name].max()

  # Calculate the number of decimal places needed
  decimals = len(str(int(max_val)))

  # Normalize the column using the decimal scaling method
  data[col_name] = data[col_name] / (10 ** decimals)

  # Print the normalized column
  return data


def encode_platform(data):
  # map the platform column to integer values
  platform_map = {'udemy': -1, 'coursera': -2, 'edx': 0, 'coding ninja': 1, 'linkedin learning': 2,'edx': 2}
  data['certification_platform'] = data['certification_platform'].replace(platform_map)

  # print the updated DataFrame
  return data


def encode_social_media(data):
  # map the platform column to integer values
  platform_map = {'youtube': -2, 'instagram': -1, 'facebook': 0, 'linkedin': 1, 'whatsapp': 2}
  data['social_media'] = data['social_media'].replace(platform_map)

  # print the updated DataFrame
  return data
  

def get_clean_data():
  data = pd.read_csv("data/Student_Data_1000.csv")
  
  marks=['cgpa','sgpa','10th_percentage','12th_percentage']
  
  for i in marks:
    data=normalize(data,i)
    
  data=encode_platform(data)
  
  data=encode_social_media(data)
  
  
  data
  
  return data


def main():
  student_data=get_clean_data()
  
  model,scaler=create_model(student_data)
  
  with open('model/model_xgb.pkl', 'wb') as f:
    pick.dump(model, f)
    
  with open('model/scaler_xgb.pkl', 'wb') as f:
    pick.dump(scaler, f)
  
  
  
  
if __name__ == '__main__':
  main()