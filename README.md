# Sentiment-Analysis-of-Movie-Reviews
🎬 Sentiment Analysis of Movie Reviews
This open-source project uses Natural Language Processing (NLP) and Machine Learning to analyze IMDB movie reviews and classify them as positive or negative. The model leverages a robust preprocessing pipeline and multiple classifiers, with Support Vector Machine (SVM) achieving the highest accuracy.

📌 Project Overview
With the rise of user-generated content, sentiment analysis helps extract public opinion on films, improving recommendation systems and decision-making for both creators and audiences. This project aims to:

Preprocess large volumes of raw review data

Train and evaluate multiple classification models

Visualize performance and draw insights from results

📊 Dataset Details
Source: IMDB Movie Reviews (Public dataset)

Size: 50,000 reviews (25,000 positive, 25,000 negative)

Attributes:

review: Text content of user review

sentiment: Label (positive/negative)

🧹 Text Preprocessing Steps
✅ HTML tag removal (e.g., <br>)

✅ Stopword removal (NLTK)

✅ Lemmatization (WordNetLemmatizer)

✅ Noise filtering (non-alphabetic chars, repeated letters)

✅ TF-IDF vectorization for numerical representation

🔍 Exploratory Data Analysis (EDA)
📈 Sentiment Distribution: Balanced (50/50)

☁️ Word Clouds:

Positive: "great", "amazing", "excellent"

Negative: "bad", "boring", "disappointing"

🤖 Machine Learning Models Implemented
Model	Accuracy	F1 Score
Naive Bayes	87%	0.87
Random Forest	89%	0.89
Logistic Regression	88%	0.88
Support Vector Machine (SVM)	90%	0.90
Best Performer: SVM — due to its effectiveness on high-dimensional sparse data from TF-IDF vectors.

📈 Evaluation Metrics
Accuracy

F1 Score

Confusion Matrix

ROC Curve & AUC

Visualizations help interpret the classification performance and comparison between models.

🛠️ Tools & Libraries Used
pandas, numpy – Data wrangling

nltk, re – Text cleaning

scikit-learn – ML models & evaluation

matplotlib, seaborn – Data visualization

🔮 Future Enhancements
💡 Integration of deep learning (LSTM, BERT)

🎭 Sentiment strength scaling (neutral, strong/mild)

🌐 Real-time web-based sentiment analysis

🧩 Aspect-based sentiment analysis (e.g., acting, direction)

🌍 Multi-language support using multilingual embeddings

🚀 Sample Predictions
Review: “A groundbreaking blend of stunning visuals and thrilling storytelling.”
Predicted Sentiment: ✅ Positive

Review: “A disjointed mess with flashes of potential. Suffered from an uneven plot.”
Predicted Sentiment: ❌ Negative
