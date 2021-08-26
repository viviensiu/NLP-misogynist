# NLP-misogynist project

# Background: 
The website Urban Dictionary is criticised for hosting content which are defined as "misogynistic, hate speech and racism". A group of researchers collected data from the website and labelled them as either misogynistic or non-misogynistic. 

Source: "Data set for automatic detection of online misogynistic speech" (https://www.sciencedirect.com/science/article/pii/S2352340919305773)

# Objective: 
From the data set, we will train a ML model for NLP text classification. The definitions will be classified as either misogynistic or non-misogynistic.

# Required libraries:
Pandas, Matplotlib, WordCloud, STOPWORDS, scikit-learn (sklearn).

# Methodology:
# ============
1) Download the public data set and perform data cleaning:
    - identify and clean missing values.
    - convert texts to lower cases and remove punctuations.
2) Perform EDA on cleaned data to analyse which keywords are most common for misogynistic and non-misogynistic.
3) Perform another round of data cleaning: remove stopwords. This is essential before the next step is performed.
4) Pass the data without stopwords into a TfidfVectorizer. Data is then split into training and test set.
5) Train 4 classification models using the training set: DummyClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier. For each model, calculate the F1 scores and confusion matrix using the predictions and actual target data from test set.
6) From the best model, extract the feature importance attribute to see which features contribute heavily in the classifications.
7) Based on the number of important features, adjust the number of features used in the best model's training to obtain a more efficient model.
8) Create some strings with misogynistic and non-misogynistic words, use these as unseen data and pass into the model to see if the predictions are accurate.

# Conclusion: 
# ===========
Out of the 4 ML models, the RandomForestClassifier model produces the best scores followed by the DecisionTreeClassifier model. After fine-tuning the RandomForestClassifier, the final model performs at a f1-score of 0.87 out of 1. 

# Ideas for expansion:
# ====================
- The data set used was extracted from Urban Dictionary between 1999 and 2016. Hence for better accuracy and future usage, it is suggested that the model be trained using an updated data set. 
- The data set could be expanded with additional labels such as racism.
- The methodology could be modified for other text classification projects such as identifying and flagging posts that violate community rules on forums and social media. 
- Explore using Deep Learning models for text classification.

# Useful reading:
# ===============
1) "How to Encode Text Data for Machine Learning with scikit-learn" (https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)
2) "TF-IDF Vectorizer scikit-learn" (https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a)
