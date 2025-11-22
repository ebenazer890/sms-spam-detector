import argparse
from sms_spam_detector.model import load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.pkl')
    parser.add_argument('--message', required=True)
    args = parser.parse_args()

    vec, clf = load_pipeline(args.model)
    x = vec.transform([args.message])
    pred = clf.predict(x)[0]
    probs = clf.predict_proba(x)[0]
    print(f"Message: {args.message}")
    print(f"Predicted: {pred}")
    print("Class probabilities:")
    for c, p in probs.items():
        print(f"  {c}: {p:.4f}")


if __name__ == '__main__':
    main()
import argparse
from sms_spam_detector.model import load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.pkl')
    parser.add_argument('--message', required=True)
    args = parser.parse_args()

    vec, clf = load_pipeline(args.model)
    x = vec.transform([args.message])
    pred = clf.predict(x)[0]
    probs = clf.predict_proba(x)[0]
    print(f"Message: {args.message}")
    print(f"Predicted: {pred}")
    print("Class probabilities:")
    for c, p in probs.items():
        print(f"  {c}: {p:.4f}")

if __name__ == '__main__':
    main()
