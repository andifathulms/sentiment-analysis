import pandas as pd
import re
import unicodedata
import html
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv('dataset.csv', encoding="latin-1")

# Initialize reusable components
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Optimized text preprocessing function
def preprocess_text(text):
    # Remove usernames, URLs, accents, HTML entities, and convert to lowercase
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = html.unescape(text).lower()
    
    # Tokenize and filter stopwords, numbers, and non-alphabetic characters
    tokens = [word for word in nltk.word_tokenize(text) if word.isalpha() and word not in stop_words]

    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Apply preprocessing (using .apply with better efficiency)
df['Text'] = df['Text'].apply(preprocess_text)

# Load positive and negative words
with open('positive.txt', 'r') as file:
    positive_words = set(line.strip() for line in file.readlines())

with open('negative.txt', 'r') as file:
    negative_words = set(line.strip() for line in file.readlines())

# Optimized sentiment calculation function
def calculate_sentiment(text):
    tokens = set(nltk.word_tokenize(text))
    pos_count = len(tokens & positive_words)
    neg_count = len(tokens & negative_words)
    return pos_count - neg_count

# Apply sentiment calculation
df['sentiment'] = df['Text'].apply(calculate_sentiment)

# Count sentiment categories
positive_count = (df['sentiment'] > 0).sum()
negative_count = (df['sentiment'] < 0).sum()
neutral_count = (df['sentiment'] == 0).sum()

# Create and display a WordCloud for all text
all_text = " ".join(df['Text'])
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_text)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Create WordClouds for positive and negative words
def generate_wordcloud(word_list, color_map, title):
    word_string = ' '.join(word_list)
    wordcloud = WordCloud(width=800, height=500, min_font_size=10, background_color='white',
                          colormap=color_map).generate(word_string)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)
    plt.show()

positive_words_list = [word for content in df['Text'] for word in content.split() if word in positive_words]
negative_words_list = [word for content in df.loc[df['sentiment'] < 0, 'Text'] for word in content.split() if word in negative_words]

generate_wordcloud(positive_words_list, 'Blues', 'Positive Words')
generate_wordcloud(negative_words_list, 'Reds', 'Negative Words')

# Print sentiment summary
print('Number of Positive Comments:', positive_count)
print('Number of Negative Comments:', negative_count)
print('Number of Neutral Comments:', neutral_count)

# Visualize sentiment distribution
labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive_count, negative_count, neutral_count]

plt.bar(labels, sizes, color=['blue', 'red', 'grey'])
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
