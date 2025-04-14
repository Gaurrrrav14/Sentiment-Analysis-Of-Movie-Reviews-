import pandas as pd

data = pd.read_csv('IMDB Dataset.csv')

data.head(20)

data.sample(10)

data.info()

data.duplicated().sum()

# duplicated_values = data[data['review'].duplicated()]['review']
# duplicated_values

data = data.drop_duplicates(subset=['review']).reset_index(drop=True)

data['sentiment'].value_counts()

import matplotlib.pyplot as plt

data_dist = [
    len(data[data['sentiment']=='positive']),
    len(data[data['sentiment']=='negative'])
]
labels = ['Positive','Negative']
plt.pie(data_dist, labels = labels)
plt.show

import re

def removeHTML(text):
    text = re.sub(r'<[^>]*>', '', text)
    return(text)

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords 
stop_words = stopwords.words('english') 

def removeStopWords(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return(text)

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def Lemmatizing(text):
    text = lemmatizer.lemmatize(text)
    return(text)

def removeNoise(text):
    text = re.sub(r'[^a-zA-Z\s]+', '', text) # Only keep English letters
    text = re.sub(r'(\w)\1{2,}', r'\1', text)   #Normalize Repeated Letters
    text = re.sub(r'\s{2,}', ' ', text).strip() # Removing extra spaces
    return(text)

CleanedText = []

for i in range(len(data)):
    text = data.review[i]
    text = removeHTML(text)
    text = removeStopWords(text)
    text = Lemmatizing(text)
    text = removeNoise(text)
    CleanedText.append(text)

data['CleanedText'] = CleanedText

#data = data.drop(['CleanedText'],axis=1)

data.head()

data.sample(20)

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train['CleanedText'])
X_test = vectorizer.transform(test['CleanedText'])

from sklearn.metrics import classification_report,f1_score,accuracy_score

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(X_train, train['sentiment'])

pred_y_mnb = mnb.predict(X_test)

print('Classification Report:\n')
print(classification_report(test['sentiment'], pred_y_mnb))
print('_'*100)
print('Accuracy:',accuracy_score(test['sentiment'], pred_y_mnb))
print('_'*100)
print('F1 Score:',f1_score(test['sentiment'], pred_y_mnb, average='weighted'))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0, n_jobs=-1, verbose=2)
rf.fit(X_train, train['sentiment'])

pred_y_rf = rf.predict(X_test)

print('Classification Report:\n')
print(classification_report(test['sentiment'], pred_y_rf))
print('_'*100)
print('Accuracy:',accuracy_score(test['sentiment'], pred_y_rf))
print('_'*100)
print('F1 Score:',f1_score(test['sentiment'], pred_y_rf, average='weighted'))

# Add this after the RandomForestClassifier section

from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression classifier
lr = LogisticRegression(max_iter=2000)

# Train Logistic Regression
lr.fit(X_train, train['sentiment'])

# Predict on test data
pred_y_lr = lr.predict(X_test)

# Print classification report, accuracy, and F1 score for Logistic Regression
print('Logistic Regression - Classification Report:')
print(classification_report(test['sentiment'], pred_y_lr))
print('_' * 100)
print('Logistic Regression Accuracy:', accuracy_score(test['sentiment'], pred_y_lr))
print('_' * 100)
print('Logistic Regression F1 Score:', f1_score(test['sentiment'], pred_y_lr, average='weighted'))

# Add this after the Logistic Regression section

from sklearn.svm import SVC

# Initialize SVM classifier
svm = SVC(kernel='linear', random_state=0)

# Train the SVM model
svm.fit(X_train, train['sentiment'])

# Predict on the test data
pred_y_svm = svm.predict(X_test)

# Print classification report, accuracy, and F1 score for SVM
print('SVM - Classification Report:')
print(classification_report(test['sentiment'], pred_y_svm))
print('_' * 100)
print('SVM Accuracy:', accuracy_score(test['sentiment'], pred_y_svm))
print('_' * 100)
print('SVM F1 Score:', f1_score(test['sentiment'], pred_y_svm, average='weighted'))


import numpy as np

def test_classifier(clf,input_text):
    input_text = removeHTML(input_text)
    input_text = removeStopWords(input_text)
    input_text = Lemmatizing(input_text)
    input_text = removeNoise(input_text)
    text_2_vec = vectorizer.transform(np.array([input_text]))    

    pred_y = clf.predict(text_2_vec)

    if pred_y == 'positive':
        return ('Positive.')
    else:
        return ('Negative.')

#Trained Classifiers: mnb, rf

input_text = '''Lacking coherence and depth, the 2015 "Fantastic Four" adaptation falters with weak
                characterization and a disjointed plot, failing to capture the essence of its source material.'''

print('Text: "'+input_text+'".')

print('Sentiment:',test_classifier(mnb,input_text))

input_text = '''A groundbreaking blend of stunning visuals and thrilling storytelling,
                "Jurassic Park" remains a timeless adventure that defined the blockbuster experience.'''

print('Text: "'+input_text+'".')

print('Sentiment:',test_classifier(mnb,input_text))

input_text = '''An intense, gripping saga showcasing Heath Ledgers mesmerizing performance
                as the Joker. "The Dark Knight" stands as a triumph in the superhero genre, blending action with profound storytelling.'''

print('Text: "'+input_text+'".')

print('Sentiment:',test_classifier(rf,input_text))

input_text = '''A disjointed mess with flashes of potential. "Suicide Squad" suffers from an uneven plot, 
                underdeveloped characters, and tonal inconsistencies that undermine its promising premise.'''

print('Text: "'+input_text+'".')

print('Sentiment:',test_classifier(rf,input_text))

input_text = '''Revolutionary in both visuals and narrative, "The Matrix" is a mind-bending sci-fi classic
                that redefined action cinema, setting new standards for storytelling and special effects.'''

print('Text: "'+input_text+'".')

print('Sentiment:',test_classifier(rf,input_text))

input_text = '''"Whiplash" is a tour de force that resonates deeply. Its intense narrative, fueled by stellar performances and
                an electrifying soundtrack, makes it an unforgettable cinematic experience.'''

print('Text: "'+input_text+'".')

print('Sentiment:',test_classifier(mnb,input_text))

input_text = '''"M. Night Shyamalan's "The Happening" is a misfire, featuring a disjointed plot 
                and cringe-inducing dialogue. Despite a promising premise, it falls flat, lacking both suspense and coherence.'''

print('Text: "'+input_text+'".')

print('Sentiment:',test_classifier(mnb,input_text))

# Add this after training your models and getting predictions

import matplotlib.pyplot as plt

# Model names
models = ['Naive Bayes', 'Random Forest']

# Accuracy scores for each model
accuracy_scores = [
    accuracy_score(test['sentiment'], pred_y_mnb),
    accuracy_score(test['sentiment'], pred_y_rf)
]

# Plotting the accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(models, accuracy_scores, color=['skyblue', 'lightgreen'])
plt.xlabel('Classification Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Classifiers')
plt.show()


# Add this after training and predictions for each model (Naive Bayes, Random Forest, Logistic Regression, and SVM)

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(pred_y, model_name):
    cm = confusion_matrix(test['sentiment'], pred_y, labels=['positive', 'negative'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Call this function after each model's prediction to visualize confusion matrix
plot_confusion_matrix(pred_y_mnb, 'Naive Bayes')
plot_confusion_matrix(pred_y_rf, 'Random Forest')
plot_confusion_matrix(pred_y_lr, 'Logistic Regression')
plot_confusion_matrix(pred_y_svm, 'SVM')


# Add this after evaluating the models (classification reports and accuracy)

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(pred_y, model_name):
    fpr, tpr, thresholds = roc_curve(test['sentiment'].map({'positive': 1, 'negative': 0}), 
                                      (pred_y == 'positive').astype(int))  # Convert predictions to binary (1 for 'positive', 0 for 'negative')
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Plot ROC curve for each classifier
plot_roc_curve(pred_y_mnb, 'Naive Bayes')
plot_roc_curve(pred_y_rf, 'Random Forest')
plot_roc_curve(pred_y_lr, 'Logistic Regression')
plot_roc_curve(pred_y_svm, 'SVM')


# Add this to visualize common words in positive and negative reviews
pip install wordcloud

from wordcloud import WordCloud

def plot_wordcloud(text_data, sentiment):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_data))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Reviews')
    plt.axis('off')
    plt.show()

# Plot wordcloud for positive and negative reviews
positive_reviews = data[data['sentiment'] == 'positive']['CleanedText']
negative_reviews = data[data['sentiment'] == 'negative']['CleanedText']

plot_wordcloud(positive_reviews, 'Positive')
plot_wordcloud(negative_reviews, 'Negative')




