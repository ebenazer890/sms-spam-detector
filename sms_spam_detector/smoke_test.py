import os
from train import load_csv
from sms_spam_detector.model import SimpleCountVectorizer, MultinomialNB, save_pipeline, load_pipeline


def evaluate(texts, labels):
    vec = SimpleCountVectorizer(min_df=1)
    X = vec.fit_transform(texts)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, labels)
    preds = clf.predict(X)
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    acc = correct / len(labels)
    return acc, vec, clf


def main():
    # data path relative to this file
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, 'data', 'sms_sample_20.csv')
    texts, labels = load_csv(data_path, n_samples=20)
    acc, vec, clf = evaluate(texts, labels)
    print(f"Trained and evaluated on {len(texts)} samples. Accuracy (train): {acc:.3f}")
    tmp = os.path.join(base, 'tmp_model.pkl')
    save_pipeline(tmp, vec, clf)
    vec2, clf2 = load_pipeline(tmp)
    preds = clf2.predict(vec2.transform(["Free entry to win cash now"]))
    print("Example prediction for 'Free entry to win cash now':", preds[0])
    os.remove(tmp)


if __name__ == '__main__':
    main()
import os
import tempfile
from train import load_csv
from sms_spam_detector.model import SimpleCountVectorizer, MultinomialNB, save_pipeline, load_pipeline


def evaluate(texts, labels):
    vec = SimpleCountVectorizer(min_df=1)
    X = vec.fit_transform(texts)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, labels)
    preds = clf.predict(X)
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    acc = correct / len(labels)
    return acc, vec, clf


def main():
    # build path relative to this file (project root/data/...)
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, 'data', 'sms_sample_20.csv')
    data_path = os.path.normpath(data_path)
    texts, labels = load_csv(data_path, n_samples=20)
    acc, vec, clf = evaluate(texts, labels)
    print(f"Trained and evaluated on {len(texts)} samples. Accuracy (train): {acc:.3f}")
    # save and reload
    tmp = 'tmp_model.pkl'
    save_pipeline(tmp, vec, clf)
    vec2, clf2 = load_pipeline(tmp)
    preds = clf2.predict(vec2.transform(["Free entry to win cash now"]))
    print("Example prediction for 'Free entry to win cash now':", preds[0])
    os.remove(tmp)

if __name__ == '__main__':
    main()
