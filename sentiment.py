import pandas as pd
import numpy as np
import re
import string
import nltk
import unicodedata
import html
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the dataset
df = pd.read_csv('dataset.csv', encoding="latin-1")
df.head()

# Text preprocessing function
def preprocess_text(text):
    # Remove Twitter username mentions
    text = re.sub(r'@[^\s]+', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Decode HTML entities
    text = html.unescape(text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Tokenize text and remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Remove numbers and punctuation
    tokens = [word for word in tokens if word.isalpha()]
    
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back into a single string
    return ' '.join(tokens)

# Apply text preprocessing to the content column
df['Text'] = df['Text'].apply(preprocess_text)
df.head(10)

# Load positive and negative word dictionaries
with open('positive.txt', 'r') as file:
    positive_words = [line.strip() for line in file.readlines()]

with open('negative.txt', 'r') as file:
    negative_words = [line.strip() for line in file.readlines()]

# Function to calculate sentiment score
def calculate_sentiment(text):
    tokens = nltk.word_tokenize(text)
    sentiment_score = 0
    
    for word in tokens:
        if word in positive_words:
            sentiment_score += 1
        elif word in negative_words:
            sentiment_score -= 1
            
    return sentiment_score

# Apply sentiment calculation to the dataset
df['sentiment'] = df['Text'].apply(calculate_sentiment)

# Count the number of comments for each sentiment category
positive_count = len(df[df['sentiment'] > 0])
negative_count = len(df[df['sentiment'] < 0])
neutral_count = len(df[df['sentiment'] == 0])
df.head(80)

# Create a WordCloud for the entire content
all_text = " ".join(df['Text'].values)
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_text)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Create a WordCloud for positive words
positive_words_list = []
for content in df['Text']:
    positive_words_list.extend([word for word in content.split() if word in positive_words])

positive_words_string = ' '.join(positive_words_list)
wordcloud = WordCloud(width=800, height=500, min_font_size=10, background_color='white', colormap='Blues').generate(positive_words_string)
wordcloud.recolor(color_func=lambda *args, **kwargs: '#1f77b4')  # Set text color to dark blue

plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# Create a WordCloud for negative words
negative_words_list = []
for _, row in df.iterrows():
    if row['sentiment'] < 0:
        negative_words_list.extend(row['Text'].split())

negative_words_string = ' '.join(negative_words_list)

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 100%, 50%)"  # Red color

wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=STOPWORDS,
                      min_font_size=10, color_func=color_func).generate(negative_words_string)

plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Print sentiment summary
print('Number of Positive Comments:', positive_count)
print('Number of Negative Comments:', negative_count)
print('Number of Neutral Comments:', neutral_count)

# Visualize sentiment distribution as a bar plot
labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive_count, negative_count, neutral_count]

plt.bar(labels, sizes, color=['blue', 'red', 'grey'])
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
