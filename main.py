import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def create_model(data):
  X = data.drop(['diagnosis'],axis=1)
  y = data['diagnosis'] 

  scalar = StandardScaler()
  X = scalar.fit_transform(X)

  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)

  model = LogisticRegression()

  model.fit(X_train,y_train)

  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  print(accuracy)

  return model,scalar




def get_clean_data():
  data = pd.read_csv('data\data.csv')
  data = data.drop(['Unnamed: 32','id'],axis=1)

  data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
  return data


def main():
  data = get_clean_data()
  
  model,scalar = create_model(data)

  with open('model.pkl','wb') as f:
    pickle.dump(model,f)
  with open('scalar.pkl','wb') as f:
    pickle.dump(scalar,f)


if __name__ == '__main__':
  main()