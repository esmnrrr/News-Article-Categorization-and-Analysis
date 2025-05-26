from preprocessing import clean_text

def predict_category(model, vectorizer, news_text):
    """ 
    Args:
        model: Eğitilmiş sınıflandırma modeli
        vectorizer: Eğitilmiş TF-IDF vectorizer
        news_text: Kategorisi tahmin edilecek haber metni (str)
        
    Returns:
        str: Tahmin edilen kategori
    """
    # Temizleme
    cleaned = clean_text(news_text)
    
    # Özellik çıkarımı
    vector = vectorizer.transform([cleaned])
    
    # Tahmin
    prediction = model.predict(vector)[0]
    
    return prediction
