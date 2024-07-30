import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('Titanic-Dataset.csv')



df['Age'].fillna(df['Age'].median(),inplace=True)

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])


X = df.drop(['Survived','Name','Ticket','Cabin'],axis=1).values
y = df['Survived'].values


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.9,random_state=42)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
#print(y_pred)
#print(df.isnull().sum())

