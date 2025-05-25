import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance

# Gerekli NLTK verilerini indir (yalnızca ilk sefer)
nltk.download('words')

# İngilizce sözlüğü al
vocabulary = set(words.words())

def correct_typos(word, vocabulary, max_distance=1):
    if word in vocabulary:
        return word, []

    closest_word = word
    min_dist = float('inf')

    for vocab_word in vocabulary:
        dist = edit_distance(word, vocab_word)
        if dist < min_dist and dist <= max_distance:
            min_dist = dist
            closest_word = vocab_word

    if closest_word != word:
        return closest_word, [(word, closest_word)]
    else:
        return word, []

# 👇 Burada denemek istediğin kelimeyi yaz
input_word = "recieve"

corrected, changes = correct_typos(input_word, vocabulary)

print("🔍 Orijinal:", input_word)
print("✅ Düzeltildi :", corrected)
if changes:
    print("✏️ Yapılan düzeltme:", changes[0][0], "➡️", changes[0][1])
else:
    print("👌 Düzeltme gerekmedi.")
# Bu kod, verilen bir kelimeyi NLTK'nin İngilizce sözlüğü ile karşılaştırarak
# en yakın kelimeyi bulur ve gerekiyorsa düzeltir.