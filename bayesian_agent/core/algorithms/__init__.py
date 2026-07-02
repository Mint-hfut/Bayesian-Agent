"""Bayesian belief update algorithms."""

from bayesian_agent.core.algorithms.beta_bernoulli import BetaBernoulliState
from bayesian_agent.core.algorithms.categorical_bayes import CategoricalBayesState, NaiveBayesState, features_from_event
from bayesian_agent.core.algorithms.hstg import HSTGState

CATEGORICAL_BAYES = "categorical_bayes"
NAIVE_BAYES_ALIAS = "naive_bayes"
BETA_BERNOULLI = "beta_bernoulli"
FREQUENTIST = "frequentist"
HSTG = "hstg"
DEFAULT_ALGORITHM = CATEGORICAL_BAYES
SUPPORTED_ALGORITHMS = (CATEGORICAL_BAYES, NAIVE_BAYES_ALIAS, BETA_BERNOULLI, FREQUENTIST, HSTG)


def normalize_algorithm(algorithm: str = None) -> str:
    if algorithm in {CATEGORICAL_BAYES, NAIVE_BAYES_ALIAS}:
        return CATEGORICAL_BAYES
    if algorithm == BETA_BERNOULLI:
        return BETA_BERNOULLI
    if algorithm == FREQUENTIST:
        return FREQUENTIST
    if algorithm == HSTG:
        return HSTG
    return DEFAULT_ALGORITHM


def is_categorical_bayes(algorithm: str = None) -> bool:
    return normalize_algorithm(algorithm) == CATEGORICAL_BAYES


def is_frequentist(algorithm: str = None) -> bool:
    return normalize_algorithm(algorithm) == FREQUENTIST


def is_hstg(algorithm: str = None) -> bool:
    return normalize_algorithm(algorithm) == HSTG

__all__ = [
    "BETA_BERNOULLI",
    "BetaBernoulliState",
    "CATEGORICAL_BAYES",
    "CategoricalBayesState",
    "DEFAULT_ALGORITHM",
    "FREQUENTIST",
    "HSTG",
    "HSTGState",
    "NAIVE_BAYES_ALIAS",
    "NaiveBayesState",
    "SUPPORTED_ALGORITHMS",
    "features_from_event",
    "is_categorical_bayes",
    "is_frequentist",
    "is_hstg",
    "normalize_algorithm",
]
