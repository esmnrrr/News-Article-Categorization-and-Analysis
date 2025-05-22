# Temizlik ve metin işleme fonksiyonlarılarını içeren modül (Normalization, lowercasing, stemming, lemmatization)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.metrics import edit_distance

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Küçük harfe çevir
    text = text.lower()

    # Özel karakterleri ve sayıları kaldır
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize et
    words = nltk.word_tokenize(text)

    # Stopword temizliği ve lemmatization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Geri birleştir
    return ' '.join(words)

# Hatalı kelimeyi düzelt
# Edit distance(Levenshtein Distance) ile en yakın kelimeyi bul
def correct_typos(text, vocabulary, max_distance=1, return_changes=False):
    #voxabulary: doğru kelimelerin listesi
    # max_distance: en fazla 1 hatası olan kelimeleri düzelt
    # text: düzeltilecek metin
    corrected_words = []
    changes = []
    for word in text.split():
        if word in vocabulary: # Eğer kelime sözlükte varsa, düzeltmeye gerek yok
            corrected_words.append(word)
        else:
            closest_word = word
            min_dist = float('inf')
            for vocab_word in vocabulary:
                dist = edit_distance(word, vocab_word)
                if dist < min_dist and dist <= max_distance:
                    min_dist = dist
                    closest_word = vocab_word
            corrected_words.append(closest_word)
            if closest_word != word:
                changes.append((word, closest_word))

    if return_changes:
        return ' '.join(corrected_words), changes
    else:
        return ' '.join(corrected_words)


# The above code defines a function to clean text data by converting it to lowercase, removing special characters and numbers,
# tokenizing the text, removing stopwords, and applying lemmatization. It then applies this function to the 'text' column of a DataFrame
