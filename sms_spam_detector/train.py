import csv
import argparse
from sms_spam_detector.model import SimpleCountVectorizer, MultinomialNB, save_pipeline


def load_csv(path, n_samples=None):
    texts = []
    labels = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if n_samples and i >= n_samples:
                break
            labels.append(row['label'])
            texts.append(row['text'])
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/sms_sample_20.csv')
    parser.add_argument('--model', default='model.pkl')
    parser.add_argument('--n', type=int, default=20, help='number of samples to use from the dataset')
    args = parser.parse_args()

    texts, labels = load_csv(args.data, n_samples=args.n)
    vec = SimpleCountVectorizer(min_df=1)
    X = vec.fit_transform(texts)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, labels)
    save_pipeline(args.model, vec, clf)
    print(f"Trained on {len(texts)} samples. Model saved to {args.model}")


if __name__ == '__main__':
    main()
import csv
import argparse
from sms_spam_detector.model import SimpleCountVectorizer, MultinomialNB, save_pipeline


def load_csv(path, n_samples=None):
    texts = []
    labels = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if n_samples and i >= n_samples:
                break
            labels.append(row['label'])
            texts.append(row['text'])
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/sms_sample_20.csv')
    parser.add_argument('--model', default='model.pkl')
    parser.add_argument('--n', type=int, default=20, help='number of samples to use from the dataset')
    args = parser.parse_args()

    texts, labels = load_csv(args.data, n_samples=args.n)
    vec = SimpleCountVectorizer(min_df=1)
    X = vec.fit_transform(texts)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, labels)
    save_pipeline(args.model, vec, clf)
    print(f"Trained on {len(texts)} samples. Model saved to {args.model}")

if __name__ == '__main__':
    main()
