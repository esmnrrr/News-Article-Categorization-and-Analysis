import os
import pandas as pd

# Kategorilerin klasör olarak bulunduğu dizin
DATASET_DIR = "datasets/bbc"

# CSV'ye yazmak için liste
data = []

# Her klasör bir kategori
for category in os.listdir(DATASET_DIR):
    category_path = os.path.join(DATASET_DIR, category)
    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                text = f.read().replace('\n', ' ')  # Satır sonlarını temizle
                data.append({'category': category, 'text': text})

# DataFrame oluştur
df = pd.DataFrame(data)

# CSV olarak kaydet
output_path = "datasets/bbc-text.csv"
df.to_csv(output_path, index=False)
print(f"CSV dosyası oluşturuldu: {output_path}")
