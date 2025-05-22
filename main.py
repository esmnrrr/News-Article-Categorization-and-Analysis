import os
import pandas as pd
from preprocessing import clean_text, correct_typos
from feature_extraction import get_tfidf_features
from classification import train_and_evaluate

# Veriyi yükle
#df = load_bbc_dataset('datasets/bbc')
df = pd.read_csv("datasets/bbc-text.csv")

# Temizleme
df['cleaned_text'] = df['text'].apply(clean_text)

# Yazım düzeltme için vocab oluştur
import nltk
nltk.download('words')
from nltk.corpus import words

vocabulary = set(words.words())

comparison_df = df[['text']].copy()
comparison_df['original_cleaned'] = df['text'].apply(clean_text)

all_changes = []

# Temizlenmiş metinlerin üzerinden geçerek düzeltmeleri uygula
corrected_texts = []
for doc in comparison_df['text']:
    corrected_raw, changes = correct_typos(doc, vocabulary, return_changes=True)
    
    # Ardından düzeltmiş metni clean_text() ile temizle
    cleaned_corrected = clean_text(corrected_raw)
    
    corrected_texts.append(cleaned_corrected)
    all_changes.extend(changes)

# DataFrame'e güncel metinleri kaydet
df['cleaned_text'] = corrected_texts

# Sadece benzersiz düzeltmeleri yazdır
unique_changes = list(set(all_changes))
print("\n✏️ Düzeltilen kelimeler (hatalı -> doğru):")
for wrong, correct in unique_changes:
    print(f"{wrong} ➡️ {correct}")

# Yazım düzeltme sırasında yapılan değişiklikleri say
num_changes = sum(1 for orig, corrected in zip(comparison_df['original_cleaned'], comparison_df['cleaned_text']) if orig != corrected)
print(f"\n🧠 Toplamda {num_changes} metinde yazım düzeltmesi yapıldı.")

# TF-IDF özellikleri
X, vectorizer = get_tfidf_features(df['cleaned_text'])

# Etiketler
y = df['category']

# Modeli eğit ve değerlendir
print("🤖 Model eğitiliyor ve değerlendiriliyor...")
model = train_and_evaluate(X, y)

def load_bbc_dataset(folder_path):
    categories = os.listdir(folder_path)
    data = []

    for category in categories:
        category_path = os.path.join(folder_path, category)
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            with open(file_path, 'r', encoding='latin1') as f:
                text = f.read()
                data.append({'category': category, 'text': text})
    
    return pd.DataFrame(data)

df = load_bbc_dataset('datasets/bbc')
print(df.head())
# The above code loads the BBC dataset from a specified folder path, where each category is stored in a separate subfolder.
# It reads the text files and creates a DataFrame with two columns: 'category' and 'text'.