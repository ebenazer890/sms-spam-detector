import csv
import argparse
from sms_spam_detector.model import SimpleCountVectorizer, MultinomialNB, save_pipeline


def load_csv(path, n_samples=None):
    """Load dataset with flexible headers and tolerant encoding.

    Supports common SMS spam CSVs that use either label/text or v1/v2 columns
    and reads with latin-1 to avoid decode failures from legacy encodings.
    """
    texts = []
    labels = []
    with open(path, newline='', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if n_samples and i >= n_samples:
                break

            # Handle both "label/text" and "v1/v2" style headers
            if 'label' in row and 'text' in row:
                label_key, text_key = 'label', 'text'
            elif 'v1' in row and 'v2' in row:
                label_key, text_key = 'v1', 'v2'
            else:
                keys = list(row.keys())
                if len(keys) < 2:
                    continue
                label_key, text_key = keys[0], keys[1]

            labels.append(row[label_key])
            texts.append(row[text_key])
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
