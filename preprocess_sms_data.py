from collections import Counter, defaultdict
import os
import csv
import random
import string
# 
BASE_DIR = './bbc';
# LABELS = ['business', 'entertainment', 'politics', 'sport', 'tech']
LABELS = ['ham', 'spam']

# def create_dataset():
#     with open('bbc_news.csv', 'w', encoding='utf8') as outfile:
#         for label in LABELS:
#             dir = '%s/%s' % (BASE_DIR, label)
#             for filename in os.listdir(dir):
#                 fullfilename = '%s/%s' % (dir, filename)
#                 print(fullfilename)
#                 with open(fullfilename, 'rb') as file:
#                     text = file.read().decode(errors='replace').replace('\n', '')
#                     outfile.write('%s\t%s\t%s\n' % (label, filename, text))

# extract the dataset 
def setup_dataset():
    data = [] #(label, text)
    with open('sms_dataset.csv', 'r') as datafile:
        reader = csv.reader(datafile)
        for row in reader:
            doc = (row[0], row[1].strip())
            data.append(doc)
    return data[1:]

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
        # data_tokens = [word.lower() for word in word_tokenize(data[1]) if word.lower() not in stop_words]
        data_tokens = [word.lower() for word in word_tokenize(data[1]) if word.lower()]
        tokens[data_label].extend(data_tokens)
    
    frequency_distribution = {} 
    for category_label, category_tokens in tokens.items():
        counter = dict(Counter(category_tokens))
        counter_sorted = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        frequency_distribution[category_label] = counter_sorted
        
    # print(frequency_distribution)
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
    
    return X_train, y_train, X_test, y_test





if __name__ == '__main__':
    
    print(get_token_distribution(setup_dataset())['spam'])
    
    dataset = setup_dataset()
    print(dataset[-1][0], len(dataset))
    ham_counter = 0
    spam_counter = len(dataset)
    for label, _ in dataset:
        if label == LABELS[0]:
            ham_counter += 1
            spam_counter -= 1
    
    print(ham_counter, spam_counter)