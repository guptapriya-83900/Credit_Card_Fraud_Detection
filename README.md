# Credit_Card_Fraud_Detection
Credit card fraud detection (CCFD) is a challenging problem, which requires analyzing large volumes of transaction data to identify fraud patterns. It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

## Goal:
Learning efficient and accurate ML models for detecting frauds in massive streams of transactions as an early warning system by using Python 3.8 on Google Colab to detect whether a transaction is a normal payment or a fraud.

## Project will tackle:

* Outlier Analysis (Identify Rare data).
* Dealing with imblanced & skewed class distributions.
* Detect Fraud Patterns with ML Models:
    * Logistic Regression.
    * KNN.
    * DecisionTree.
    * Random Forest.
* ML metrics for model validation.
* Precision-Recall tradeoff.

## Dataset Overview:
* The dataset contains transactions made by credit cards in September 2013 by European cardholders.
* Due to confidentiality issues, the original features and more background information about the data are not available.
* This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
* It contains only numerical input variables which are the result of a PCA transformation.
* Dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Data Visualization
* Univariante Analysis.
* Bivariante Analysis.
* Multivariante Analysis.
* Questions Answers.

## Result:
* We investigated the data, checking for data unbalancing, visualizing the features and understanding the relationship between different features. 
* Implemented SMOTE on our imbalanced dataset helped us with the imbalance of our labels (more no fraud than fraud transactions).
* Removal of outliers was not implemented which can be done in future work and see if that will affect our model performance.
* After comparison between models, we found that *Logistic Regression Classifier* with SMOTE gave us the best result.
