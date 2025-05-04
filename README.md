# Phase 3 Code Challenge

This assessment is designed to test your understanding of Module 3 material. It covers:

* Gradient Descent
* Logistic Regression
* Classification Metrics
* Decision Trees

_Read the instructions carefully_. You will be asked both to write code and to answer short answer questions.

## Code Tests

We have provided some code tests for you to run to check that your work meets the item specifications. Passing these tests does not necessarily mean that you have gotten the item correct - there are additional hidden tests. However, if any of the tests do not pass, this tells you that your code is incorrect and needs changes to meet the specification. To determine what the issue is, read the comments in the code test cells, the error message you receive, and the item instructions.

## Short Answer Questions 

For the short answer questions...

* _Use your own words_. It is OK to refer to outside resources when crafting your response, but _do not copy text from another source_.

* _Communicate clearly_. We are not grading your writing skills, but you can only receive full credit if your teacher is able to fully understand your response. 

* _Be concise_. You should be able to answer most short answer questions in a sentence or two. Writing unnecessarily long answers increases the risk of you being unclear or saying something incorrect.

```python
# Run this cell without changes to import the necessary libraries
import pickle, sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from numbers import Number
```

---
## Part 1: Gradient Descent [Suggested Time: 20 min]
---
In this part, you will describe how gradient descent works to calculate a parameter estimate. Below is an image of a best fit line from a linear regression model using TV advertising spending to predict product sales.

![best fit line](https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_3/best_fit_line.png)

This best fit line can be described by the equation $y = mx + b$. Below is the RSS cost curve associated with the slope parameter $m$:

![cost curve](https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_3/cost_curve.png)

where RSS is the residual sum of squares: $RSS = \sum_{i=1}^n(y_i - (mx_i + b))^2$ 

### 1.1) Short Answer: Explain how the RSS curve above could be used to find an optimal value for the slope parameter $m$. Your answer should provide a one sentence summary, not every step of the process.


# Your answer here # brianwaweru-answer-1.1
`The RSS curve could be used to find the ideal value for the slope parameter m by determinign the point where the curve reaches its minimum, indicating the value of m that results in the least residual error between the predicted and actual values.`

Below is a visualization showing the iterations of a gradient descent algorithm applied on the RSS curve. Each yellow marker represents an estimate, and the lines between markers represent the steps taken between estimates in each iteration. Numeric labels identify the iteration numbers.

![gradient descent](https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_3/gd.png)


### 1.2) Short Answer: Explain why the distances between markers get smaller over successive iterations.

#### Your answer here # brianwaweru-answer-2
`The distances between the markers get smaller each time because, as gradient descent gets closer to the best value for the slope m, the changes it makes get smaller meaning it's getting closer to the lowest point on the cost curve.`

### 1.3) Short Answer: What would be the effect of decreasing the learning rate for this application of gradient descent?

#### Your answer here # brianwaweru-answer-3
`If we lower the learning rate, gradient descent will move more slowly toward the ideal answer, but it will be more precise and less likely to go past it or make mistakes.`

---
## Part 2: Logistic Regression [Suggested Time: 15 min]
---

In this part, you will answer general questions about logistic regression.

### 2.1) Short Answer: Provide one reason why logistic regression is better than linear regression for modeling a binary target/outcome.

#### Your answer here # brianwaweru-answer-2.1
`Logistic regression is better because it gives results between 0 and 1, which helps us decide if something should be labeled as one group or the other making classification much easier.`

### 2.2) Short Answer: Compare logistic regression to another classification model of your choice (e.g. Decision Tree). What is one advantage and one disadvantage logistic regression has when compared with the other model?


#### Your answer here # brianwaweru-answer-2.2
`One advantage of logistic regression is that its less prone to overfitting due to its simplicity and fewer parameters as compared to Decision-Trees. However, one disadvantage is its low-flexibility than Decision-Trees in capturing non-linear relationships because of its assumption of linear decision boundary.`

---
## Part 3: Classification Metrics [Suggested Time: 20 min]
---

In this part, you will make sense of classification metrics produced by various classifiers.
The confusion matrix below represents the predictions generated by a classisification model on a small testing dataset.

![cnf matrix](https://curriculum-content.s3.amazonaws.com/data-science/images/cnf_matrix.png)

### 3.1) Create a numeric variable `precision` containing the precision of the classifier.

```python
# CodeGrade step3.1
# Replace None with appropriate code ## brianwaweru-answer-3.1

precision = 30 / (30 + 4) # rem precision = TP / (TP + FP) # 0.8823529411764706
```

```python
# This test confirms that you have created a numeric variable named precision

assert isinstance(precision, Number)
```

### 3.2) Create a numeric variable `f1score` containing the F-1 score of the classifier.

```python
# CodeGrade step3.2
# Replace None with appropriate code ## brianwaweru-answer-3.2

f1score = 2 * (precision * (30 / (30 + 12))) / (precision + (30 / (30 + 12))) # 0.7894736842105262
```

```python
# This test confirms that you have created a numeric variable named f1score

assert isinstance(f1score, Number)
```

The ROC curves below were calculated for three different models applied to one dataset.

1. Only Age was used as a feature in the model
2. Only Estimated Salary was used as a feature in the model
3. All features were used in the model

![roc](https://curriculum-content.s3.amazonaws.com/data-science/images/many_roc.png)

### 3.3) Short Answer: Identify the best ROC curve in the above graph and explain why it is the best. 

#### Your answer here ## brianwaweru-answer-3.3
`The Model that included All Features is the best since it appears to have the highest AUC.`

Run the following cells to load a sample dataset, run a classification model on it, and perform some EDA.

```python
# Run this cell without changes
network_df = pickle.load(open('sample_network_data.pkl', 'rb'))

# partion features and target 
X = network_df.drop('Purchased', axis=1)
y = network_df['Purchased']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2019)

# scale features
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

# build classifier
model = LogisticRegression(C=1e5, solver='lbfgs')
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

# get the accuracy score
print(f'The classifier has an accuracy score of {round(accuracy_score(y_test, y_test_pred), 3)}.')
# Run this cell without changes

y.value_counts()
```

### 3.4) Short Answer: Explain how the distribution of `y` shown above could explain the high accuracy score of the classification model.

#### Your answer here ## brianwaweru-answer-3.4
`The data is highly imbalanced: most people didn’t purchase (257) and only a few did (13). The model would get high accuracy just by predicting “no purchase” most of the time, but that wouldn’t mean it’s actually good at identifying real purchases. This raises a concern that accuracy isn’t a reliable measure in this case.`

### 3.5) Short Answer: What is one method you could use to improve your model to address the issue discovered in Question 3.4?

#### Your answer here ## brianwaweru-answer-3.5
`One way to improve the model is by using class weighting i.e. by setting class_weight='balanced' paramter in the model which assigns greater importance to the minority class, helping the model focus more on correctly classifying it during training.`

---
## Part 4: Decision Trees [Suggested Time: 20 min]
---

In this part, you will use decision trees to fit a classification model to a wine dataset. The data contain the results of a chemical analysis of wines grown in one region in Italy using three different cultivars (grape types). There are thirteen features from the measurements taken, and the wines are classified by cultivar in the `target` variable.

```python
# Run this cell without changes
# Relevant imports 
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier

# Load the data 
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'target'
```

### 4.1) Use `train_test_split()` to split `X` and `y` data between training sets (`X_train` and `y_train`) and test sets (`X_test` and `y_test`), with `random_state=1`. Evenly split the data between train and test (50/50).

Do not alter `X` or `y` before performing the split.

```python
# CodeGrade step4.1
# Replace None with appropriate code ## brianwaweru-answer-4.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
```

```python
# These tests confirm that you have created DataFrames named X_train, X_test and Series named y_train, and y_test

assert type(X_train) == pd.DataFrame
assert type(X_test) == pd.DataFrame
assert type(y_train) == pd.Series
assert type(y_test) == pd.Series
```

```python
# These tests confirm that you have split the data evenly between train and test sets

assert X_train.shape[0] == X_test.shape[0]
assert y_train.shape[0] == y_test.shape[0]
```

### 4.2) Create an untuned decision tree classifier `wine_dt` with `random_state=1` and fit it using `X_train` and `y_train`. 

Use parameter defaults for your classifier. You must use the Scikit-learn DecisionTreeClassifier (docs [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))


```python
# CodeGrade step4.2
# Replace None with appropriate code ## brianwaweru-answer-4.2

wine_dt = DecisionTreeClassifier(random_state=1)

# Fit
wine_dt.fit(X_train, y_train)
```

```python
# This test confirms that you have created a DecisionTreeClassifier named wine_dt

assert type(wine_dt) == DecisionTreeClassifier

# This test confirms that you have set random_state to 1

assert wine_dt.get_params()['random_state'] == 1

# This test confirms that wine_dt has been fit

sklearn.utils.validation.check_is_fitted(wine_dt)
```

Create an array `y_pred` generated by using `wine_dt` to make predictions for the test data.### 4.3) 

```python
# CodeGrade step4.3
# Replace None with appropriate code ## brianwaweru-answer-4.3

y_pred = wine_dt.predict(X_test)
```

```python
# This test confirms that you have created an array-like object named y_pred

assert type(np.asarray(y_pred)) == np.ndarray
```

### 4.4) Create a numeric variable `wine_dt_acc` containing the accuracy score for your predictions. 

Hint: You can use the `sklearn.metrics` module or the model itself.

```python
# CodeGrade step4.4
# Replace None with appropriate code ## brianwaweru-answer-4.4
from sklearn.metrics import accuracy_score
wine_dt_acc = accuracy_score(y_test, y_pred)
```

```python
# This test confirms that you have created a numeric variable named wine_dt_acc

assert isinstance(wine_dt_acc, Number)
```

### 4.5) Short Answer: Based on the accuracy score, does the model seem to be performing well or does it have substantial performance issues? Explain your answer.

#### Your answer here ## brianwaweru-answer-4.5
`The model seems to be performing relatively well, with an accuracy score of approximately 87.6% which suggests that the model is able to correctly predict the wine cultivar in about 87.6% of the cases on the test data.`