"""Bayesian belief update algorithms."""

from bayesian_agent.core.algorithms.beta_bernoulli import BetaBernoulliState
from bayesian_agent.core.algorithms.naive_bayes import NaiveBayesState, features_from_event

DEFAULT_ALGORITHM = "naive_bayes"
SUPPORTED_ALGORITHMS = ("naive_bayes", "beta_bernoulli")

__all__ = [
    "BetaBernoulliState",
    "DEFAULT_ALGORITHM",
    "NaiveBayesState",
    "SUPPORTED_ALGORITHMS",
    "features_from_event",
]
