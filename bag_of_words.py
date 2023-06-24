import math
from LogisticRegression import LogisticRegression
from NaiveBayes import NaiveBayes
from preprocess_email_data import *
 
# Feature extraction 1: Bag of words (collection of words)

LABELS = ['ham', 'spam']

dataset = create_dataset()

test_dataset, training_set, training_labels, testing_set, testing_labels = get_splits(dataset)

distribution = get_token_distribution(test_dataset[: int(0.8 * len(test_dataset))])



# Select the most indicative bag of words
def select_a_bag_of_words(bag_weight = 500):
    words = []
    repeated = []
    words_per_category = {}
    for category, token_distribution in distribution.items():
        i = 0
        per_category = []
        for word in token_distribution:
            if i < int(bag_weight/2):
                if word not in words:
                    words.append(word)
                    per_category.append(word)
                    i += 1
                elif word in words and word not in repeated:
                    repeated.append(word)
            else:
                break
        words_per_category[category] = per_category

    bag = []
    for category, word_tokens in words_per_category.items():
        word_tokens = [word for word in word_tokens if word not in repeated]
        bag.extend(word_tokens[:int(bag_weight/5)])

    return bag


bag_of_words = select_a_bag_of_words()


# Vectorize an article as per the bag
def vectorize(article, bag=bag_of_words):
    vector = [0] * len(bag)
    for i, word in enumerate(bag):
        vector[i] += article.count(word)
    return vector


# Vectorize a whole training dataset
def vectorize_list(data_list):
    vector_set = []
    for article in data_list:
        vector = vectorize(article, bag_of_words)
        vector_set.append(vector)
    return vector_set


def naive(learning_rate, laplace_smoothing):
        X_train = vectorize_list(training_set)
        X_test = vectorize_list(testing_set)

        y_train = [LABELS.index(label) for label in training_labels]
        y_test = [LABELS.index(label) for label in testing_labels]

        nb = NaiveBayes(learning_rate=learning_rate, alpha=laplace_smoothing)
        
        nb.fit(X_train, y_train)
        result = nb.predict(X_test)

        labeled_wrongly = []
        i = 0
        while i < len(y_test):
            if y_test[i] != result[i]:
                labeled_wrongly.append((y_test[i], result[i]))
            i += 1
        
        accuracy = 1 - ((len(labeled_wrongly))/(len(y_test)))
        print(accuracy)
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
        print(accuracy)
        return accuracy * 100


if __name__ == '__main__':
    LearningRates = [ 0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
    LaplaceSmoothing =[ 0.1, 0.5, 1.0, 10, 100]

    def get_results_naive():
        print("Naive Bayes bag of words")
        for lr in LearningRates:
            for ls in LaplaceSmoothing:
                acc = naive(learning_rate=lr, laplace_smoothing=ls)
                print("learning rate: ", lr,"\t", "smoothing: ", ls,"\t", "accuracy: ", acc)
                
    def get_results_logistic():
        print("Logistic Regressions bag of words")
        for lr in LearningRates:
            acc = logistic(lr)
            print("learning rate: ", lr,"\t", "accuracy: ", acc)
    
    # get_results_naive()
    # get_results_logistic()
    
    # naive(1.0, 0.1)
    # nb = NaiveBayes(learning_rate=1.0, alpha=1)
    
    logistic(0.1)