from LogisticRegression import LogisticRegression
from NaiveBayes import NaiveBayes
from preprocess_sms_data import *

# Feature extraction 2: using Bigrams

LABELS = ['ham', 'spam']

dataset = setup_dataset()
training_set, training_labels, testing_set, testing_labels = get_splits(dataset)
# X_train, y_train, X_test, y_test


# Tokenize the sets and convert to bigrams
def tokenize_to_bigrams(x_set, n_sentences=1):  
    return [to_bigram(word_tokenize(text)) for text in x_set]

def to_bigram(unigrams):
    bigrams = []
    for i in range(len(unigrams) - 1):
        bigram = unigrams[i] + ' ' + unigrams[i+1]
        bigrams.append(bigram)
    return bigrams


bigram_X_train = tokenize_to_bigrams(training_set)



def bigram_distribution(datalist):
    bi_tokens = defaultdict(list)
    
    for data in datalist:
        data_label = data[0]
        data_tokens = list(set([bigram for bigram in to_bigram(word_tokenize(data[1]))]))
        bi_tokens[data_label].extend(data_tokens)

    frequency_distribution = {} 
    for category_label, category_tokens in bi_tokens.items():
        counter = dict(Counter(category_tokens))
        counter_sorted = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        frequency_distribution[category_label] = counter_sorted
        
    # print(frequency_distribution)
    return frequency_distribution
    
bi_distribution = bigram_distribution(dataset)


# Select the most indicative bag of words
def select_vocabulary(max_length = 1000):
    bigrams = []
    repeated = []
    bi_per_category = {}
    for category, token_distribution in bi_distribution.items():
        i = 0
        per_category = []
        for word in token_distribution:
            if i < int(max_length/2):
                if word not in bigrams:
                    bigrams.append(word)
                    per_category.append(word)
                    i += 1
                elif word in bigrams and word not in repeated:
                    repeated.append(word)
            else:
                break
        bi_per_category[category] = per_category

    bag = []
    for category, bi_tokens in bi_per_category.items():
        bi_tokens = [word for word in bi_tokens if word not in repeated]
        bag.extend(bi_tokens[:int(max_length/5)])

    return bag



vocabulary = select_vocabulary()
# print(vocabulary, len(vocabulary))


def vectorize(document, vocabulary=vocabulary):
    vector = []
    for bigram in vocabulary:
        if bigram in document:
            vector.append(1)  # Bigram present in the document
        else:
            vector.append(0)
    return vector

def vectorize_list(data, vocabulary=vocabulary):
    feature_matrix = []
    for document in data:
        vector = vectorize(document, vocabulary=vocabulary)
        feature_matrix.append(vector)
    return feature_matrix

# X_train = vectorize(training_set, vocabulary)
# print(X_train, len(X_train), len(X_train[0]))


def naive(learning_rate= 1.0, laplace_smoothing = 1.0):
        X_train = vectorize_list(training_set)
        X_test = vectorize_list(testing_set)

        y_train = [LABELS.index(label) for label in training_labels]
        y_test = [LABELS.index(label) for label in testing_labels]

        nb = NaiveBayes(learning_rate, laplace_smoothing)
        
        nb.fit(X_train, y_train)
        result = nb.predict(X_test)

        labeled_wrongly = []
        i = 0
        while i < len(y_test):
            if y_test[i] != result[i]:
                labeled_wrongly.append((y_test[i], result[i]))
            i += 1
        
        accuracy = 1 - ((len(labeled_wrongly))/(len(y_test)))
        # print(accuracy)
        return accuracy * 100
        
def logistic(learning_rate):
        X_train = vectorize_list(training_set)
        X_test = vectorize_list(testing_set)

        y_train = [LABELS.index(label) for label in training_labels]
        y_test = [LABELS.index(label) for label in testing_labels]
        
        lr = LogisticRegression(learning_rate=learning_rate)
        lr.fit(X_train, y_train)

        result = lr.predict(X_test)

        labeled_wrongly = []
        i = 0
        while i < len(y_test):
            if y_test[i] != result[i]:
                labeled_wrongly.append((y_test[i], result[i]))
            i += 1
        
        accuracy = 1 - ((len(labeled_wrongly))/(len(y_test)))
        # print(accuracy)
        return accuracy * 100

if __name__ == '__main__':
    LearningRates = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5]
    LaplaceSmoothing =[ 0.1, 0.5, 1.0, 10, 100]

    def get_results_naive():
        print("Naive Bayes bigrams")
        for lr in LearningRates:
            for ls in LaplaceSmoothing:
                acc = naive(learning_rate=lr, laplace_smoothing=ls)
                print("learning rate: ", lr,"\t", "smoothing: ", ls,"\t", "accuracy: ", acc)
                
    def get_results_logistic():
        print("Logistic Regressions bigrams")
        for lr in LearningRates:
            acc = logistic(lr)
            print("learning rate: ", lr,"\t", "accuracy: ", acc)
    
    # get_results_naive()
    get_results_logistic()
    text = 'free online money making'
    # text = 'We are trying to contact you. URGENT We are trying to contact you Last weekends draw shows u have won a £1000 prize GUARANTEED Call 09064017295 Claim code K52 Valid 12hrs 150p pm'
    # print(change_to_vector(text))
    
    X_train = vectorize_list(training_set)
    X_test = vectorize_list(testing_set)

    y_train = [LABELS.index(label) for label in training_labels]
    y_test = [LABELS.index(label) for label in testing_labels]
    
    train = []
    train.extend(X_train)
    train.extend(X_test)
    test = []
    test.extend(y_train)
    test.extend(y_test)
    
    logReg = LogisticRegression(learning_rate=0.1)
    logReg.fit(train, test)
    predict = logReg.predict([vectorize(text)])
    print(predict)
    # # print(train)