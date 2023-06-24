import csv
import os
import random
import time
import string
from collections import Counter, defaultdict, namedtuple

BASE_DIR = './data_archive';
LABELS = ['ham', 'spam']

def create_dataset():
    dataset = []
    
    count = 0
    punctuation = string.punctuation
    
    with open('email_dataset.csv', 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['label', 'filename', 'text'])
        
        for i in range(1, 7):
            curr_dir = '%s/%s' % (BASE_DIR, f'enron{i}')
            
            for label in LABELS:
                dir = '%s/%s' % (curr_dir, label)
                
                for filename in os.listdir(dir):
                    fullfilename = '%s/%s' % (dir, filename)
                    count += 1
                    # print(fullfilename, count)
                    
                    with open(fullfilename, 'rb') as file:
                        text = file.read().decode(errors='replace').replace('\n', '').replace('\t', '')
                        text = ''.join([char for char in text if char not in punctuation])
                        if len(text) > 131072:
                            text = text[:131072]
                        writer.writerow([label, filename, text])
                        dataset.append((label, str(text.encode(encoding='utf-8'))))
    return dataset                  

def word_tokenize(text):
    text = text.strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return tokens


stop_words = []
with open('stopwords.csv', 'r', encoding='utf8') as stopwords:
    for word in stopwords:
        stop_words.append(word.strip())


def get_token_distribution(data_list):
    tokens = defaultdict(list)
    
    for data in data_list:
        data_label = data[0]
        data_tokens = [word.lower() for word in word_tokenize(data[1]) if word.lower() not in stop_words]
        tokens[data_label].extend(data_tokens)
    
    frequency_distribution = {} 
    for category_label, category_tokens in tokens.items():
        counter = dict(Counter(category_tokens))
        counter_sorted = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        frequency_distribution[category_label] = counter_sorted
        
    print(frequency_distribution)
    return frequency_distribution


def get_token_distribution_allowing_stopwords(data_list):
    tokens = defaultdict(list)
    
    for data in data_list:
        data_label = data[0]
        data_tokens = [word.lower() for word in word_tokenize(data[1]) if word.lower()]
        tokens[data_label].extend(data_tokens)
    
    frequency_distribution = {} 
    for category_label, category_tokens in tokens.items():
        counter = dict(Counter(category_tokens))
        counter_sorted = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        frequency_distribution[category_label] = counter_sorted
        
    # print(frequency_distribution)
    return frequency_distribution

# split dataset into test and train
def get_splits(dataset):
    # shuffle the dataset
    random.shuffle(dataset)
    
    X_train = [] # training articles
    y_train = [] # corresponding lables
    
    X_test = [] # testing articles
    y_test = [] # testing lables

    pivot = int(0.80 * len(dataset))
    # 80% of the dataset is for training and the rest for testing
    
    X_train.extend([data[1] for data in dataset[:pivot]]) 
    y_train.extend([data[0] for data in dataset[:pivot]]) 

    X_test.extend([data[1] for data in dataset[pivot:]]) 
    y_test.extend([data[0] for data in dataset[pivot:]]) 
    
    return dataset, X_train, y_train, X_test, y_test



            
if __name__ == '__main__':
    start_time = time.time()
    get_token_distribution(create_dataset())
            
            
    print(f'elapsed time: {time.time() - start_time} seconds')