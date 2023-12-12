from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib 

# Read original dataset
iris_data = datasets.load_iris()
#iris_df.sample(frac=1, random_state=seed)
# selecting features and target data
iris_df = pd.DataFrame(data=iris_data['data'],columns=iris_data['feature_names'])
iris_df['iris_type'] = iris_data['target']
X = iris_df[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]]
y = iris_df[['iris_type']]
# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30, stratify=y)
# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
# train the classifier on the training data
clf.fit(X_train, y_train)
# predict on the test set
y_pred = clf.predict(X_test)
# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}") # Accuracy: 0.91

joblib.dump(clf, "iris_model.sav")