import re
import math


def get_words(doc):
    splitter = re.compile('\\W+')
    words = [s.lower() for s in splitter.split(doc) if 2 < len(s) < 20]
    return dict([(w,1) for w in words])


class Classifier:

    def __init__(self, get_features):
        # feature/category counter
        self.fc = {}
        # category counter
        self.cc = {}
        self.thresholds = {}
        self.get_features = get_features

    def in_cf(self, f, cat):
        self.fc.setdefault(f,{})
        self.fc[f].setdefault(cat,0)
        self.fc[f][cat] += 1

    def in_cc(self, cat):
        self.cc.setdefault(cat,0)
        self.cc[cat] += 1

    # feature count
    def f_count(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    # category count
    def cat_count(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    # total count
    def total_count(self):
        return sum(self.cc.values())

    # categories list
    def categories(self):
        return self.cc.keys()

    # learning train
    def train(self, item, cat):
        features = self.get_features(item)
        for f in features:
            self.in_cf(f, cat)
        self.in_cc(cat)

    def f_prob(self, f, cat):
        if self.cat_count(cat) == 0:
            return 0
        # word probability in class
        return self.f_count(f, cat) / self.cat_count(cat)

    def weighted_prob(self, f, cat, prf, weight=1.0, ap=0.5):
        # computing assumed probability for rare words:
        basic_prob = prf(f, cat)
        totals = sum([self.f_count(f, c) for c in self.categories()])
        bp = (weight * ap + totals * basic_prob) / (weight + totals)
        return bp

    def sample_train(self):
        self.train('Nobody owns the water.','good')
        self.train('the quick rabbit jumps fences','good')
        self.train('buy pharmaceuticals now','bad')
        self.train('make quick money at the online casino','bad')
        self.train('the quick brown fox jumps','good')

    def set_tr(self, cat, t):
        self.thresholds[cat] = t

    def get_tr(self, cat):
        if cat not in self.thresholds:
            return 3
        return self.thresholds[cat]


class NaiveBayes(Classifier):

    def doc_prob(self, item, cat):
        features = self.get_features(item)
        p = 1
        for f in features:
            p *= self.weighted_prob(f, cat, self.f_prob)
        return p

    def prob(self, item, cat):
        cat_prob = self.cat_count(cat) / self.total_count()
        doc_prob = self.doc_prob(item ,cat)
        return cat_prob * doc_prob

    def classify(self, item, default='Unknown'):
        probs = {}
        max = 0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat
        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.get_tr(best) > probs[best]:
                return default
        return best

cl = NaiveBayes(get_words)
cl.sample_train()
print(cl.classify('quick rabbit'))
print(cl.classify('quick money'))

########################################################################


class Fisher(Classifier):

    def __init__(self, get_features):
        Classifier.__init__(self, get_features)
        self.minimums = {}

    def c_prob(self, f, cat):
        # feature freq in this category
        clf = self.f_prob(f, cat)
        if clf == 0:
            return 0
        # feature freq in all categories
        freq_sum = sum([self.f_prob(f, c) for c in self.categories()])
        p = clf / freq_sum
        return p

    def fisher_prob(self, item, cat):
        p = 1
        features = self.get_features(item)
        for f in features:
            p *= (self.weighted_prob(f, cat, self.c_prob))
        f_score = -2 * math.log(p)
        return self.inv_chi2(f_score, len(features) * 2)

    @staticmethod
    def inv_chi2(chi, df):
        m = chi / 2
        sum = term = math.exp(-m)
        for i in range(1, df // 2 ):
            term *= m / i
            sum += term
        return min(sum, 1)

    def set_minimum(self, cat, min):
        self.minimums[cat] = min

    def get_minimum(self, cat):
        if cat not in self.minimums:
            return 0
        return float(self.minimums[cat])

    def classify(self, item, default='Unknown'):
        best = default
        max = 0.0
        for c in self.categories():
            p = self.fisher_prob(item, c)
            if p > self.get_minimum(c) and p > max:
                best = c
                max = p
        return best

cl = Fisher(get_words)
cl.sample_train()

# print(cl.fisher_prob('quick rabbit', 'good'))
print(cl.classify('quick rabbit'))
print(cl.classify('quick money'))
cl.set_minimum('bad','0.8')
print(cl.classify('quick money'))
cl.set_minimum('good','0.5')
print(cl.classify('quick money'))

