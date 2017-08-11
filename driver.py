import preprocessing
import d2v
import classifier

from gensim.models.doc2vec import Doc2Vec

txts = []

# used the enron datasets found here:
# http://www2.aueb.gr/users/ion/data/enron-spam/
for s in range(1, 6):
    txts += preprocessing.load_files('data/enron%i/ham/' % s, False)
    txts += preprocessing.load_files('data/enron%i/spam/' % s, True)

split_index = int(len(txts) / 5.0)
testing_txts = txts[:split_index]
training_txts = txts[split_index:]

# d2v_model = d2v.train_d2v_model(training_txts)
d2v_model = Doc2Vec.load('models/d2v.model')

# for the classifier half
train_xs, train_ys = d2v.txt_2_vectors(training_txts, d2v_model, True)
test_xs, test_ys = d2v.txt_2_vectors(testing_txts, d2v_model, False)

classifier.train_classifier(train_xs, train_ys, test_xs, test_ys)
