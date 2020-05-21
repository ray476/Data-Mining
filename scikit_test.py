import os
import sys
import operator
import time
import numpy as np
import contractions
import knn
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')  # needed by word_tokenize
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
# list of words found to still rank very high (10k+) after a running with just stopwords and examining vocab.txt
other_words = ['lines', 'subject']

# Assuming this file is put under the same parent directoray as the data directory, and the data directory is named "20news-train"
root_path = "./20news-train"
# The maximum size of the final vocabulary. It's a hyper-parameter. You can change it to see what value gives the best performance.
MAX_VOCAB_SIZE = 5000

start_time = time.time()
vocab_full = {}
n_doc = 0
# Only keep the data dictionaries and ignore possible system files like .DS_Store
folders = [os.path.join(root_path, name) for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
for folder in folders:
    for filename in os.listdir(folder):
        file = os.path.join(folder, filename)
        n_doc += 1
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                # split contractions into two words
                line = contractions.fix(line)
                tokens = word_tokenize(line)
                # force everything to lower case and remove non-alphabetic characters
                tokens = [token.lower() for token in tokens if token.isalpha()]
                for token in tokens:
                    # remove stop words, other words (above) and single characters
                    if (token not in stop_words) and (token not in other_words) and (len(token) > 1):
                        vocab_full[token] = vocab_full.get(token, 0) + 1
print(f'{n_doc} documents in total with a total vocab size of {len(vocab_full)}')
vocab_sorted = sorted(vocab_full.items(), key=operator.itemgetter(1), reverse=True)
vocab_truncated = vocab_sorted[:MAX_VOCAB_SIZE]
# Save the vocabulary to file for visual inspection and possible analysis
with open('vocab1.txt', 'w') as f:
    for vocab, freq in vocab_truncated:
        f.write(f'{vocab}\t{freq}\n')
# The final vocabulary is a dict mapping each token to its id. frequency information is not needed anymore.
vocab = dict([(token, id) for id, (token, _) in enumerate(vocab_truncated)])
# Since we have truncated the vocabulary, we will encounter many tokens that are not in the vocabulary. We will map all of them to the same 'UNK' token (a common practice in text processing), so we append it to the end of the vocabulary.
vocab['UNK'] = MAX_VOCAB_SIZE
vocab_size = len(vocab)
unk_id = MAX_VOCAB_SIZE
elapsed_time = time.time() - start_time
print(f'Vocabulary construction took {elapsed_time} seconds')

# Since we have truncated the vocabulary, it's now reasonable to hold the entire feature matrix in memory (it takes about 3.6GB on a 64-bit machine). If memory is an issue, you could make the vocabulary even smaller or use sparse matrix.
start_time = time.time()
features = np.zeros((n_doc, vocab_size), dtype=int)
print(f'The feature matrix takes {sys.getsizeof(features)} Bytes.')
# The class label of each document
labels = np.zeros(n_doc, dtype=int)
# The mapping from the name of each class label (i.e., the subdictionary name corresponding to a topic) to an integer ID
label2id = {}
label_id = 0
doc_id = 0
for folder in folders:
    label2id[folder] = label_id
    for filename in os.listdir(folder):
        labels[doc_id] = label_id
        file = os.path.join(folder, filename)
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                tokens = word_tokenize(line)
                for token in tokens:
                    # if the current token is in the vocabulary, get its ID; otherwise, get the ID of the UNK token
                    token_id = vocab.get(token, unk_id)
                    features[doc_id, token_id] += 1
        doc_id += 1
    label_id += 1
elapsed_time = time.time() - start_time
print(f'Feature extraction took {elapsed_time} seconds')

tree = SklearnDecisionTreeClassifier(max_depth=10)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
tree.fit(x_train, y_train)
predictions = list()
for test_row in x_test:
    predictions.append(tree.predict(test_row.reshape(1, -1)))
labels_f = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
export_graphviz(
            tree,
            out_file="tree.dot",
            #feature_names=vocab_truncated,
            class_names=labels_f,
            rounded=True,
            filled=True,
        )
correct = 0
i = 0
for label in predictions:
    if label[0] == y_test[i]:
        correct += 1
score = correct / float(len(y_test)) * 100
print('{} correct predictions were made for a score of {}%'.format(correct, score))