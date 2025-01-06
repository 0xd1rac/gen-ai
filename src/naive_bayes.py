import numpy as np 
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.classes = None

        # prior probability of each class - P(C_k)
        self.class_prior = {}

        # conditional probability of each feature given each class (P(X_i | C_k))
        self.likelihoods = defaultdict(lambda: defaultdict(float)) 

        # set of all unique words in the training data
        self.vocabulary = set()

    def fit(self, X, y):
        """
        Fit the Naive Bayes model to training data
        :param X: list of documents (preprocessed as lists of words)
        :param y: list of class labels
        """
        self.classes = np.unique(y)
        class_counts = defaultdict(int) # A defaultdict(int) automatically initializes missing keys with a default value of 0.
        word_counts = defaultdict(lambda: defaultdict(int))

        for doc,label in zip(X,y):
            class_counts[label] += 1
            for word in doc:
                self.vocabulary.add(word)
                word_counts[label][word] += 1

        # Calculate prior probabilities and likelihoods
        total_docs = len(X)
        for label in self.classes:
            # P(C_k) = num_docs_in_class C_k / total_docs
            self.class_prior[label] = class_counts[label] / total_docs

        # Calculate likelihoods with Laplace smoothing (add 1 to avoid zero division)
        # P(X_i | C_k) = (count(X_i, C_k) + 1) / (total_words_in_class C_k + num_unique_words)
        for label in self.classes:
            total_words = sum(word_counts[label].values())
            for word in self.vocabulary:
                self.likelihoods[label][word] = (
                    (word_counts[label][word] + 1) / (total_words + len(self.vocabulary))
                )

    def predict(self, X):
        """
        Predict the class for a new document
        :param X: list of documents (preprocessed as lists of words)
        :return: list of predicted class labels
        """
        predictions = []
        for doc in X:
            class_scores = {}
            for label in self.classes:
                # Start with log of prior probability - log(P(C_k))
                class_scores[label] = np.log(self.class_prior[label])

            # add the log probability of each word in the document
            for word in doc:
                if word in self.vocabulary:
                    # score(C_k | X) = log(P(C_k)) + sum(log(P(X_i | C_k)))
                    class_scores[label] += np.log(self.likelihoods[label][word])

            # predict the class with the highest score - C_hat = argmax(score(C_k | X))
            predictions.append(max(class_scores, key=class_scores.get))

        return predictions

# Example usage
if __name__ == "__main__":
    # Training data
    X_train = [
        ["buy", "cheap", "viagra"],
        ["limited", "time", "offer"],
        ["meet", "me", "for", "coffee"],
        ["urgent", "money", "needed"],
        ["free", "entry", "lottery"]
    ]
    y_train = ["spam", "spam", "ham", "spam", "spam"]
    
    # Test data
    X_test = [
        ["free", "coffee"],
        ["urgent", "money"]
    ]
    
    # Train and predict
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    print(predictions)  # Output: ['ham', 'spam']