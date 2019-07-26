import numpy as np
import pandas as pd

from nltk import pos_tag
from nltk.corpus import PlaintextCorpusReader, wordnet
from scipy.sparse import coo_matrix, hstack, csr_matrix
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold


def parse_line(line):
    """Parse a line of the lexicon file into an MpqaWord named tuple."""
    data = {}
    for pair in line.strip().split():
        try:
            key, val = pair.split("=")
            data[key] = val
        except ValueError: # Occasional errant string
            continue
    return data

# Set containing all unique positive words.
positives = set()

# Set containing all unique negative words.
negatives = set()

with open("MPQA_Subjectivity_Cues_Lexicon.tff", "r") as f:
    # For each word in the lexicon store associated information in a dictionary.
    # Words are keys, values are dictionaries containing the key-value pairs from the lexicon.
    lexicon_words = {}

    # Read, parse and extract data from all the lines in the lexicon.
    lines = f.readlines()
    for line in lines:
        current_data = parse_line(line)
        lexicon_words[current_data["word1"]] = current_data
        if current_data["priorpolarity"] == "positive":
            positives.add(current_data["word1"])
        if current_data["priorpolarity"] == "negative":
            negatives.add(current_data["word1"])


def baseline_classifier(review_tokens):
    """Baseline classifier function."""
    pos_words_count = 0
    neg_words_count = 0

    # Count the no. of positive and negative words in the review according to the lexicon.
    for token in review_tokens:
        if token in positives:
            pos_words_count += 1
        elif token in negatives:
            neg_words_count += 1

    # Make a decicion.
    # Neutral reviews with the same no. of positive and negative words are ignored.
    if pos_words_count > neg_words_count:
        return 1
    elif pos_words_count < neg_words_count:
        return 0

# Save all reviews in a list.
all_reviews = []

# Save the corresponding polarity labels in a list for all reviews.
labels = []

with open("rt-polaritydata/rt-polarity.pos", "r") as f:
    """Read the positive reviews from the given file."""
    lines = f.readlines()

    all_reviews.extend(lines)
    labels.extend(["POSITIVE" for el in range(0, len(lines))])

    correctly_classified_positive_reviews = 0
    total_positive_reviews_no = len(lines)

    for line in lines:
        # Tokenize current review.
        raw_tokens = line.split(" ")
        tokens = [t.strip(" -()\\/'\"!?.&") for t in raw_tokens]

        # Classify review using the baseline classifier.
        result = baseline_classifier(tokens)

        # Count correct classifications.
        if result == 1:
            correctly_classified_positive_reviews += 1

# Print accuracy of classification for positive reviews.
positives_accuracy = 100*correctly_classified_positive_reviews/float(total_positive_reviews_no)
print("\nPositive Reviews Baseline Classifier Accuracy: " + str(positives_accuracy))


with open("rt-polaritydata/rt-polarity.neg", "r") as f:
    """Read the negative reviews from the given file."""
    lines = f.readlines()

    all_reviews.extend(lines)
    labels.extend(["NEGATIVE" for el in range(0, len(lines))])

    correctly_classified_negative_reviews = 0
    total_negative_reviews_no = len(lines)

    for line in lines:
        # Tokenize current review.
        raw_tokens = line.split(" ")
        tokens = [t.strip(" -()\\/'\"!?.&") for t in raw_tokens]

        # Classify review using the baseline classifier.
        result = baseline_classifier(tokens)

        # Count correct classifications.
        if result == 0:
            correctly_classified_negative_reviews += 1

# Print accuracy of classification for negative reviews.
negatives_accuracy = 100*correctly_classified_negative_reviews/float(total_negative_reviews_no)
print("Negative Reviews Baseline Classifier Accuracy: " + str(negatives_accuracy))

# Print Baseline Classifier average accuracy of classification.
print("Average Baseline Classifier Accuracy: " + str((positives_accuracy+negatives_accuracy)/2.0))

# Store for each review how many positive and negative words it contains as features.
sentiment_features = []

for review in all_reviews:
    positive_words = 0
    negative_words = 0
    
    # Tokenize current review.
    raw_tokens = review.split(" ")
    tokens = [token.strip(" -()\\/'\"!?.&") for token in raw_tokens]

    # Count the positive and negative words according to the lexicon.
    for token in tokens:
        if token in positives:
            positive_words += 1
        elif token in negatives:
            negative_words += 1

    # Store the positive and negative no. of words as features to be later used in the classifier.
    sentiment_features.append([positive_words, negative_words])


# Process and vectorize the reviews to a numeric feature matrix containing 
# the tf-idf scores for all the words in the corpus as features + the no. 
# of positive and negative sentiment words in the review according to the
# lexicon as features.

data = {"reviews": all_reviews, "sentiment": labels}
reviews_corpus = pd.DataFrame(data=data)

# Vectorize the reviews to tf-idf values of all the words in the corpus.
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(reviews_corpus["reviews"])
X = vectorizer.transform(reviews_corpus["reviews"]).todense()

# Append the no. of positive and negative words in the reviews as features to the feature matrix.
sentiment_features = np.array(sentiment_features)
X = np.column_stack((X, sentiment_features[:, 0]))
X = np.column_stack((X, sentiment_features[:, 1]))
X = csr_matrix(X)

# Instantiate an SVM classifier with default hyperparameters and linear kernel.
svm_classifier = svm.SVC(kernel="linear", gamma="scale")

# Perform cross-validation with 5 splits.
cv = KFold(n_splits=5, shuffle=True)
acc_scores = []
for train_index, test_index in cv.split(X):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = reviews_corpus["sentiment"][train_index]
    y_test = reviews_corpus["sentiment"][test_index]

    # Train classifier.
    svm_classifier.fit(X_train, y_train)

    # Store test accuracies for each fold in order to computer their mean.
    acc_scores.append(svm_classifier.score(X_test, y_test))

print("\nMean Accuracy for SVM Classifier (k-fold validation: 5 splits): " + str(100*np.mean(acc_scores)) + "\n")
