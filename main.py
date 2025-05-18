import os
import pandas as pd

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