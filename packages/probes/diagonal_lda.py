import numpy as np 
from functools import partial
class DiagonalLDA():
    """Diagonal LDA for two class classification
    """
    def __init__(self, m1=None, m2=None, pev=None, classes=None, class_priors=[]):
        self.centroid_one = m1
        self.centroid_two = m2
        self.pev = pev
        self.classes_ = classes
        # self.class_priors = class_priors
        self.class_priors = [0.5, 0.5]

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)

        x1s = X[y==1] 
        x2s = X[y==0]

        m1 = x1s.mean(axis=0)
        m2 = x2s.mean(axis=0)

        ev1 = ((x1s - m1)**2).sum(axis=0) 
        ev2 =((x2s - m2)**2).sum(axis=0) 
        pev = (ev1 + ev2) / (len(X) - 2)
        # pev = np.ones(m1.shape)

        self.centroid_one = m1
        self.centroid_two = m2
        self.pev = pev
        # self.class_priors = [len(x2s)/len(X), len(x1s)/len(X)]
        self.class_priors = [0.5, 0.5]
        return self

    def predict(self, X):
        m1, m2, pev = self.centroid_one, self.centroid_two, self.pev
        dc = lambda mc, class_prior, vec: -(((vec - mc)**2)/2*pev).sum() + np.log(class_prior)


        prob_c1 = np.apply_along_axis(partial(dc, m1, self.class_priors[1]), 1, X)
        prob_c2 = np.apply_along_axis(partial(dc, m2, self.class_priors[0]), 1, X)
        
        probs = np.column_stack((prob_c2, prob_c1,))
        preds = probs.argmax(axis=1) 
        return self.classes_[preds]

    def predict_log_proba(self, X):
        """Return the log probability DENSITY of 
        belonging to the two classes.
        For the two-class problem, the problem is equivalent to 
        Equivalent to \delta_{c}(x) = log p(x,y=c|\theta). That is, 
        it is a joint probability; not a direct probability of p(y=c).
        See page 108 (Chapter 4) of Murphy's probablistic learning.
        """
        m1, m2, pev = self.centroid_one, self.centroid_two, self.pev 
        # dc = lambda mc, vec: -(((vec - mc)**2) / 2*(pev)**2).sum()

        dc = lambda mc, vec: -(((vec - mc)**2)/(2*(pev**2))).sum() 
        prob_c1 = np.apply_along_axis(partial(dc, m1), 1, X)
        prob_c2 = np.apply_along_axis(partial(dc, m2), 1, X)
        
        probs = np.column_stack((prob_c1, prob_c2))
        return probs

    def get_params(self, deep=False):
        return {}