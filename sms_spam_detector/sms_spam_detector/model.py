import re
import math
import pickle
from collections import defaultdict, Counter


class SimpleCountVectorizer:
    """Very small tokenizer + count vectorizer.
    - lowercases, removes non-alphanum, splits on whitespace
    - supports fit, transform, fit_transform
    """
    def __init__(self, min_df=1):
        self.min_df = min_df
        self.vocabulary_ = {}

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if t]
        return tokens

    def fit(self, documents):
        df = Counter()
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df[t] += 1
        idx = 0
        for t, count in df.items():
            if count >= self.min_df:
                self.vocabulary_[t] = idx
                idx += 1
        return self

    def transform(self, documents):
        rows = []
        for doc in documents:
            cnt = defaultdict(int)
            for t in self._tokenize(doc):
                if t in self.vocabulary_:
                    cnt[self.vocabulary_[t]] += 1
            rows.append(dict(cnt))
        return rows

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


class MultinomialNB:
    """Simple Multinomial Naive Bayes with add-one smoothing."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_count_ = Counter()
        self.feature_count_ = defaultdict(Counter)
        self.class_log_prior_ = {}
        self.feature_log_prob_ = defaultdict(dict)
        self.classes_ = []
        self.n_features_ = 0

    def fit(self, X_counts, y):
        self.classes_ = sorted(set(y))
        self.n_features_ = 0
        for row in X_counts:
            for idx in row.keys():
                if idx + 1 > self.n_features_:
                    self.n_features_ = idx + 1
        for row, label in zip(X_counts, y):
            self.class_count_[label] += 1
            for idx, c in row.items():
                self.feature_count_[label][idx] += c
        total = sum(self.class_count_.values())
        for c in self.classes_:
            self.class_log_prior_[c] = math.log(self.class_count_[c] / total)
        for c in self.classes_:
            total_count = sum(self.feature_count_[c].values())
            denom = total_count + self.alpha * self.n_features_
            for idx in range(self.n_features_):
                num = self.feature_count_[c].get(idx, 0.0) + self.alpha
                self.feature_log_prob_[c][idx] = math.log(num / denom)
        return self

    def predict_log_proba_single(self, x_counts):
        res = {}
        for c in self.classes_:
            logp = self.class_log_prior_.get(c, float('-inf'))
            for idx, cnt in x_counts.items():
                if idx < self.n_features_:
                    logp += self.feature_log_prob_[c].get(idx, math.log(self.alpha / (self.alpha * self.n_features_))) * cnt
                else:
                    logp += math.log(self.alpha / (self.alpha * (self.n_features_ + 1))) * cnt
            res[c] = logp
        return res

    def predict(self, X_counts):
        preds = []
        for row in X_counts:
            logp = self.predict_log_proba_single(row)
            preds.append(max(logp.items(), key=lambda x: x[1])[0])
        return preds

    def predict_proba(self, X_counts):
        out = []
        for row in X_counts:
            logp = self.predict_log_proba_single(row)
            vals = list(logp.values())
            maxv = max(vals)
            exps = [math.exp(v - maxv) for v in vals]
            s = sum(exps)
            probs = [e / s for e in exps]
            out.append({c: p for c, p in zip(self.classes_, probs)})
        return out


# helpers to save/load pipeline
def save_pipeline(path, vectorizer, model):
    model_state = {
        'alpha': model.alpha,
        'class_count': dict(model.class_count_),
        'feature_count': {k: dict(v) for k, v in model.feature_count_.items()},
        'class_log_prior': dict(model.class_log_prior_),
        'feature_log_prob': {k: dict(v) for k, v in model.feature_log_prob_.items()},
        'classes': list(model.classes_),
        'n_features': int(model.n_features_),
    }
    with open(path, 'wb') as f:
        pickle.dump({'vocab': vectorizer.vocabulary_, 'model_state': model_state}, f)


def load_pipeline(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    vec = SimpleCountVectorizer()
    vec.vocabulary_ = data['vocab']
    state = data['model_state']
    model = MultinomialNB(alpha=state.get('alpha', 1.0))
    model.class_count_ = Counter(state.get('class_count', {}))
    model.feature_count_ = defaultdict(Counter, {k: Counter(v) for k, v in state.get('feature_count', {}).items()})
    model.class_log_prior_ = dict(state.get('class_log_prior', {}))
    model.feature_log_prob_ = defaultdict(dict, {k: dict(v) for k, v in state.get('feature_log_prob', {}).items()})
    model.classes_ = list(state.get('classes', []))
    model.n_features_ = int(state.get('n_features', 0))
    return vec, model

import re
import math
import pickle
from collections import defaultdict, Counter

class SimpleCountVectorizer:
    """Very small tokenizer + count vectorizer.
    - lowercases, removes non-alphanum, splits on whitespace
    - supports fit, transform, fit_transform
    """
    def __init__(self, min_df=1):
        self.min_df = min_df
        self.vocabulary_ = {}
        self._idf = {}

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if t]
        return tokens

    def fit(self, documents):
        df = Counter()
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df[t] += 1
        # keep tokens with df >= min_df
        idx = 0
        for t, count in df.items():
            if count >= self.min_df:
                self.vocabulary_[t] = idx
                idx += 1
        return self

    def transform(self, documents):
        # return list of dicts: index -> count
        rows = []
        for doc in documents:
            cnt = defaultdict(int)
            for t in self._tokenize(doc):
                if t in self.vocabulary_:
                    cnt[self.vocabulary_[t]] += 1
            rows.append(dict(cnt))
        return rows

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

class MultinomialNB:
    """Simple Multinomial Naive Bayes with add-one smoothing."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_count_ = Counter()
        # use Counter as the default factory (module-level callable) so pickling works
        self.feature_count_ = defaultdict(Counter)
        self.class_log_prior_ = {}
        self.feature_log_prob_ = defaultdict(dict)
        self.classes_ = []
        self.n_features_ = 0

    def fit(self, X_counts, y):
        # X_counts: list of dict(index->count)
        self.classes_ = sorted(set(y))
        self.n_features_ = 0
        for row in X_counts:
            for idx in row.keys():
                if idx + 1 > self.n_features_:
                    self.n_features_ = idx + 1
        # count features per class
        for row, label in zip(X_counts, y):
            self.class_count_[label] += 1
            for idx, c in row.items():
                self.feature_count_[label][idx] += c
        # compute log priors
        total = sum(self.class_count_.values())
        for c in self.classes_:
            self.class_log_prior_[c] = math.log(self.class_count_[c] / total)
        # compute feature log prob with smoothing
        for c in self.classes_:
            # total count of all features for class
            total_count = sum(self.feature_count_[c].values())
            denom = total_count + self.alpha * self.n_features_
            for idx in range(self.n_features_):
                num = self.feature_count_[c].get(idx, 0.0) + self.alpha
                self.feature_log_prob_[c][idx] = math.log(num / denom)
        return self

    def predict_log_proba_single(self, x_counts):
        # returns dict class -> log-prob
        res = {}
        for c in self.classes_:
            logp = self.class_log_prior_.get(c, float('-inf'))
            for idx, cnt in x_counts.items():
                if idx < self.n_features_:
                    logp += self.feature_log_prob_[c].get(idx, math.log(self.alpha / (self.alpha * self.n_features_))) * cnt
                else:
                    # unseen feature index -> use uniform smoothing
                    logp += math.log(self.alpha / (self.alpha * (self.n_features_ + 1))) * cnt
            res[c] = logp
        return res

    def predict(self, X_counts):
        preds = []
        for row in X_counts:
            logp = self.predict_log_proba_single(row)
            preds.append(max(logp.items(), key=lambda x: x[1])[0])
        return preds

    def predict_proba(self, X_counts):
        out = []
        for row in X_counts:
            logp = self.predict_log_proba_single(row)
            # softmax
            vals = list(logp.values())
            maxv = max(vals)
            exps = [math.exp(v - maxv) for v in vals]
            s = sum(exps)
            probs = [e / s for e in exps]
            out.append({c: p for c, p in zip(self.classes_, probs)})
        return out

# helpers to save/load pipeline

def save_pipeline(path, vectorizer, model):
    # Serialize only primitive structures to avoid pickling callables
    model_state = {
        'alpha': model.alpha,
        'class_count': dict(model.class_count_),
        'feature_count': {k: dict(v) for k, v in model.feature_count_.items()},
        'class_log_prior': dict(model.class_log_prior_),
        'feature_log_prob': {k: dict(v) for k, v in model.feature_log_prob_.items()},
        'classes': list(model.classes_),
        'n_features': int(model.n_features_),
    }
    with open(path, 'wb') as f:
        pickle.dump({'vocab': vectorizer.vocabulary_, 'model_state': model_state}, f)


def load_pipeline(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    vec = SimpleCountVectorizer()
    vec.vocabulary_ = data['vocab']
    state = data['model_state']
    model = MultinomialNB(alpha=state.get('alpha', 1.0))
    model.class_count_ = Counter(state.get('class_count', {}))
    model.feature_count_ = defaultdict(Counter, {k: Counter(v) for k, v in state.get('feature_count', {}).items()})
    model.class_log_prior_ = dict(state.get('class_log_prior', {}))
    model.feature_log_prob_ = defaultdict(dict, {k: dict(v) for k, v in state.get('feature_log_prob', {}).items()})
    model.classes_ = list(state.get('classes', []))
    model.n_features_ = int(state.get('n_features', 0))
    return vec, model
