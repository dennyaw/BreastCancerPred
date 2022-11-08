#import data
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
features = pd.read_csv('/breast-cancer-wisconsin.data')

#Import library
import pandas as pd
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

#Change the columns name and delete ID
features.columns = ['Id', 'Clump_thickness', 'Uniformity_cell_size', 'Uniformity_cell_shape', 'Marginal_adhesion', 'Single_e_cell_size', 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses', 'Class']
features.drop('Id',axis=1,inplace=True)

#Preprocessing data
features.drop(features[features['Bare_nuclei']=='?'].index,inplace=True)
features.astype({'Bare_nuclei': 'int64'}).dtypes

# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(features['Class'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Class', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

#n_estimators 100
#will be changed to 200-500

#timer start
start_time = time.time()

n = 10
while (n < 100):
    train = n/100
    test = 1-train
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=train, test_size = test)

    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 100 decision trees
    rf = RandomForestClassifier(n_estimators = 100)
    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    
    #Calculate the accuracy
    print(accuracy_score(test_labels,predictions))
    
    n=n+10
    
#timer stop
print("Time taken: %s seconds \n" % (time.time() - start_time))

#n_estimators 200

#timer start
start_time = time.time()

n = 10
while (n < 100):
    train = n/100
    test = 1-train
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    #train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=train, test_size = test, random_state = 42)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=train, test_size = test,random_state = 0)

    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 300 decision trees
    rf = RandomForestClassifier(n_estimators = 200,random_state = 0)
    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    
    #Calculate the accuracy
    print(accuracy_score(test_labels,predictions))
    
    n=n+10
    
#timer stop
print("Time taken: %s seconds \n" % (time.time() - start_time))    

#n_estimators 300

#timer start
start_time = time.time()
n = 10
while (n < 100):
    train = n/100
    test = 1-train
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=train, test_size = test, random_state = 42)

    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 300 decision trees
    rf = RandomForestClassifier(n_estimators = 300, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    
    #Calculate the accuracy
    print(accuracy_score(test_labels,predictions))
    
    n=n+10
    
#timer stop
print("Time taken: %s seconds \n" % (time.time() - start_time))    

#n_estimators 400

#timer start
start_time = time.time()
n = 10
while (n < 100):
    train = n/100
    test = 1-train
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=train, test_size = test, random_state = 42)

    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 400 decision trees
    rf = RandomForestClassifier(n_estimators = 400, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    
    #Calculate the accuracy
    print(accuracy_score(test_labels,predictions))
    
    n=n+10
    
#timer stop
print("Time taken: %s seconds \n" % (time.time() - start_time))    

#n_estimators 500

#timer start
start_time = time.time()
n = 10
while (n < 100):
    train = n/100
    test = 1-train
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=train, test_size = test, random_state = 42)

    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 500 decision trees
    rf = RandomForestClassifier(n_estimators = 500, random_state = 42,oob_score=True)
    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    
    #Calculate the accuracy
    print(accuracy_score(test_labels,predictions))
    
    n=n+10
    
#timer stop
print("Time taken: %s seconds \n" % (time.time() - start_time))    


#dt 90, n 300
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, train_size=0.9,random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 300 decision trees
rf = RandomForestClassifier(n_estimators = 300,random_state = 42,oob_score=True)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
    
#Calculate the accuracy
#print(accuracy_score())
from sklearn import metrics
print(metrics.classification_report(test_labels,predictions))

#dt 90, n 400
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, train_size=0.9,random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 300 decision trees
rf = RandomForestClassifier(n_estimators = 400,random_state = 42,oob_score=True)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
    
#Calculate the accuracy
#print(accuracy_score())
from sklearn import metrics
print(metrics.classification_report(test_labels,predictions))


#dt 90, n 500
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, train_size=0.9,random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 300 decision trees
rf = RandomForestClassifier(n_estimators = 500,random_state = 42,oob_score=True)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
    
#Calculate the accuracy
#print(accuracy_score())
from sklearn import metrics
print(metrics.classification_report(test_labels,predictions))


#Visualize
#dt 90, n 300
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 300, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

#confusion matrix
f,ax=plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(test_labels,predictions),annot=True,fmt=".0f",ax=ax)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

#decision tree
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[0]
# Export the image to a dot file
export_graphviz(tree, out_file = 'finaltree2.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('finaltree2.dot')
# Write graph to a png file
graph.write_png('finaltree2.png')