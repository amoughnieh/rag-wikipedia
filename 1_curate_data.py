import os
import json
from datasets import load_dataset

full_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split='train')

dataset = full_dataset.shuffle(seed=42).select(range(50000))

script_dir = os.getcwd()
data_folder = os.path.join(script_dir, 'data', 'raw_documents')

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

for article in dataset:
    article_data = {
        'id': article['id'],
        'url': article['url'],
        'title': article['title'],
        'text': article['text'],
    }
    file_path = os.path.join(data_folder, f"{article['id']}.json")
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            print(f.name, 'does not exist. creating file..')
            json.dump(article_data, f, indent=4)

if __name__ == '__main__':
    pass