# Project Backlog

## 1. EDA
Idk i'll see with christian later

## 2. Preprocessing
1. Manually remove outliers, in particular for:
- Age that is outside of range [15,56]
- Weight that is over 167
- Height is already removed with the two previous queries (see EDA)

2. Scale numerical values and impute missing numerical values with KNN and de-scale numerical values

3. Combine age, height and weight to make bmi class feature
- To ask Michal if he got his criterions from anywhere which should be included in the report

4. Fill missing activity with "No activity" as the missing values could potentially indicate no activity at all (since in the metadata 1 was the minimum)
- By experimenting with models, we obtained that by filling with "No Activity" got a better validation score

5. Encode categorical variables with ordinal values (except for gender and yes/no categories)

6. Obtain life score from the categorical variables
- By experimenting, we found out that a simple sum was the best way
    - Example: by doing weighted sums such as penalizing "bad" attributes we got a worse performance

## 3. Feature Selection
We used the following algorithms for feature selection:
1. Filter methods:
- Pearson Correlation
- According to Spearman correlation between variables
- Spearman Rank Correlation to test variables' independence
2. Wrapper methods:
- RFE concluded that 
3. Embedded methods:
- We fitted a Random Forest classifier to the train dataset and extracted its feature importances.

## 3.1. Results of FS
Pearson correlation: nothing can be said

Spearman correlation: there is a strong correlation between weight and BMI class (0.8)

Spearman Rank Correlation: following variables are important: alcohol frequency, caloric frequency, monitoring calories, parent overweight, physical activity per week, daily water consumption, bmi class, average meals per day.

The following variables has a higher 3% importance: weight, bmi class, age, height, gender and life score.

RFE: age, alcohol frequency, gender, height, weight and bmi class are important features

## 4. Model Selection
We used repeated k-fold for evaluating models and selected the model as the one with the best test score (macro average f1-score).
In particular, we have tested with the following models:
- Logistic regression with different solvers
- Simple decision tree
- Random Forest Classifier
- Gradient Boosted Decision Tree
To optimize our model, we have used GridSearch to find the best hyperparameters.