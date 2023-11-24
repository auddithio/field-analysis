"""
Filename: logistic_regression.py
--------------------------------
This program runs a simple logistic regression of the data
from the various fields.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""
The columns of the dataset are as follows:
[date, location, description, LST_diff_1km, NDVI, NDWI, VH_gamma0, VV_gamma0]

Here, description is the label, and columns after that correspond to the 
features we are prediciting off of.

The labels here are fallow, green, growing season flooding, 
ripe/harvest (yellow), winter flooded.
"""
data = pd.read_csv("data/concatenated_file.csv")

"""
First, we run a binary logistic regression.
"""
# Create a dataset with binary description labels, either flooded or not flooded
binary_data = data.copy()
labels = {"fallow": 0, "green" : 0, "ripe/harvest (yellow)": 0, "growing season flooded": 1, "growing season flooding": 1, "winter flooded": 1}

binary_data["description"] = binary_data["description"].map(labels)
binary_data.to_csv("data/binary_data.csv", index=False)


"""
Runs a simple logistic regression model
"""
def run_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:\n', conf_matrix)
    print('Classification Report:\n', classification_rep)


"""
Use a basline model: here this is just the dates
"""
# Baseline: compute temporal changes (using months)
binary_data['date'] = pd.to_datetime(binary_data["date"])
binary_data['month'] = binary_data["date"].dt.month

X = binary_data["month"]
y = binary_data["description"]

run_logistic_regression(X.to_frame().values.reshape((-1, 1)), y)

"""
Use a better feature set (this time, all the other parameters including the date), 
as a step above baseline.
"""

# For each of the features, try to model in a simple binary logistic regression
X = binary_data[["month", "LST_diff_1km", "NDVI", "NDWI", "VH_gamma0", "VV_gamma0"]]
y = binary_data["description"]

run_logistic_regression(X, y)