"""Hierarchical spatio-temporal Bayesian evidence model (HSTG).

HSTG extends the categorical Bayesian evidence model with a
task-conditioned dynamic class prior built from two shrinkage axes:

- Spatial shrinkage: historical trajectory evidence is weighted by a
  semantic kernel ``K(x_t, x_i) in [0, 1]`` between the current task text
  and each recorded task text, producing kernel-weighted local Beta counts
  ``alpha_local`` / ``beta_local``.
- Temporal shrinkage: a confidence weight ``w_local`` grows with the total
  kernel mass in the current task's neighborhood, so sparse or dissimilar
  history falls back smoothly to the global class prior.

The fused dynamic prior replaces the static Laplace class prior inside the
categorical likelihood product:

    pi'(success | x_t) = w_local * pi_local + (1 - w_local) * pi_global
    P(y | h, x_t, features) proportional to pi'(y | x_t) * prod_j P(x_j | y, h)

``w_local`` is computed from kernel mass only (pseudo-counts excluded), so
with zero similar history the model reduces exactly to the categorical
backend's behavior. This is a kernel-weighted hierarchical shrinkage
scheme, not an exact conjugate posterior: kernel weights break
exchangeability, so ``Beta(alpha_local, beta_local)`` should be read as a
local pseudo-likelihood belief.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from bayesian_agent.core.algorithms.categorical_bayes import LABELS, CategoricalBayesState
from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.core.similarity import SimilarityProvider, default_similarity


DEFAULT_ALPHA0 = 1.0
DEFAULT_BETA0 = 1.0
DEFAULT_C_STABLE = 4.0
MAX_EVENTS = 200
MAX_TASK_TEXT = 1500
_EPS = 1e-9


@dataclass
class HSTGState:
    """Kernel-weighted hierarchical belief state for one Skill hypothesis."""

    alpha0: float = DEFAULT_ALPHA0
    beta0: float = DEFAULT_BETA0
    c_stable: float = DEFAULT_C_STABLE
    categorical: CategoricalBayesState = field(default_factory=CategoricalBayesState)
    events: List[Dict[str, Any]] = field(default_factory=list)
    max_events: int = MAX_EVENTS

    @property
    def observations(self) -> int:
        return int(self.categorical.observations)

    def update(self, event: TrajectoryEvidence) -> "HSTGState":
        self.categorical.update(event)
        self.events.append(
            {
                "task_id": str(event.task_id or ""),
                "task_text": str(getattr(event, "task_text", "") or event.summary or "")[:MAX_TASK_TEXT],
                "label": "success" if event.success else "failure",
                "failure_mode": str(event.failure_mode or ""),
                "context": str(event.context or ""),
            }
        )
        self.events = self.events[-int(self.max_events) :]
        return self

    def local_evidence(self, task_text: str = "", provider: Optional[SimilarityProvider] = None) -> Dict[str, Any]:
        """Kernel-weighted success/failure mass in the current task's neighborhood."""

        provider = provider or default_similarity()
        task_text = str(task_text or "")
        success_mass = 0.0
        failure_mass = 0.0
        neighbors: List[Dict[str, Any]] = []
        if task_text:
            for item in self.events:
                text = str(item.get("task_text") or "")
                if not text:
                    continue
                kernel = max(0.0, min(1.0, float(provider.similarity(task_text, text))))
                if kernel <= 0.0:
                    continue
                if item.get("label") == "success":
                    success_mass += kernel
                else:
                    failure_mass += kernel
                neighbors.append(
                    {
                        "task_id": item.get("task_id", ""),
                        "label": item.get("label", ""),
                        "failure_mode": item.get("failure_mode", ""),
                        "kernel": round(kernel, 4),
                    }
                )
        neighbors.sort(key=lambda entry: -float(entry["kernel"]))
        return {
            "success_mass": success_mass,
            "failure_mass": failure_mass,
            "neighbors": neighbors[:5],
        }

    def dynamic_prior(self, task_text: str = "", provider: Optional[SimilarityProvider] = None) -> Dict[str, Any]:
        """Fuse the local kernel-weighted belief with the global class prior."""

        local = self.local_evidence(task_text, provider)
        success_mass = float(local["success_mass"])
        failure_mass = float(local["failure_mass"])
        alpha_local = self.alpha0 + success_mass
        beta_local = self.beta0 + failure_mass
        kernel_mass = success_mass + failure_mass
        w_local = kernel_mass / (kernel_mass + self.c_stable) if kernel_mass > 0.0 else 0.0
        pi_local = alpha_local / (alpha_local + beta_local)
        pi_global = self.categorical.class_probability("success")
        pi_success = w_local * pi_local + (1.0 - w_local) * pi_global
        return {
            "alpha_local": alpha_local,
            "beta_local": beta_local,
            "kernel_mass": kernel_mass,
            "w_local": w_local,
            "pi_local": pi_local,
            "pi_global": pi_global,
            "pi_success": pi_success,
            "neighbors": local["neighbors"],
        }

    def predict_proba(
        self,
        features: Mapping[str, Any] = None,
        task_text: str = "",
        provider: Optional[SimilarityProvider] = None,
    ) -> Dict[str, float]:
        prior = self.dynamic_prior(task_text, provider)
        pi_success = min(1.0 - _EPS, max(_EPS, float(prior["pi_success"])))
        priors = {"success": pi_success, "failure": 1.0 - pi_success}
        features = dict(features or {})
        logs = {}
        for label in LABELS:
            logs[label] = math.log(priors[label])
            for name, value in sorted(features.items()):
                logs[label] += math.log(self.categorical.feature_probability(name, value, label))
        max_log = max(logs.values())
        scores = {label: math.exp(value - max_log) for label, value in logs.items()}
        total = sum(scores.values())
        return {label: scores[label] / total if total else 0.0 for label in LABELS}

    def predict_success(
        self,
        features: Mapping[str, Any] = None,
        task_text: str = "",
        provider: Optional[SimilarityProvider] = None,
    ) -> float:
        return self.predict_proba(features, task_text=task_text, provider=provider).get("success", 0.0)

    def weighted_failure_support(
        self,
        task_text: str = "",
        provider: Optional[SimilarityProvider] = None,
    ) -> Dict[str, float]:
        """Kernel-weighted support per failure mode for patch activation.

        Without task text the weights degrade to raw failure counts, which
        reproduces the count-based gating of the other backends.
        """

        provider = provider or default_similarity()
        task_text = str(task_text or "")
        support: Dict[str, float] = {}
        for item in self.events:
            if item.get("label") == "success":
                continue
            failure_mode = str(item.get("failure_mode") or "")
            if not failure_mode:
                continue
            if task_text:
                text = str(item.get("task_text") or "")
                weight = max(0.0, min(1.0, float(provider.similarity(task_text, text)))) if text else 0.0
            else:
                weight = 1.0
            if weight <= 0.0:
                continue
            support[failure_mode] = support.get(failure_mode, 0.0) + weight
        return support

    def audit(self, task_text: str = "", provider: Optional[SimilarityProvider] = None) -> Dict[str, Any]:
        prior = self.dynamic_prior(task_text, provider)
        return {
            "observations": self.observations,
            "c_stable": float(self.c_stable),
            "alpha_local": round(float(prior["alpha_local"]), 4),
            "beta_local": round(float(prior["beta_local"]), 4),
            "kernel_mass": round(float(prior["kernel_mass"]), 4),
            "w_local": round(float(prior["w_local"]), 4),
            "pi_local": round(float(prior["pi_local"]), 4),
            "pi_global": round(float(prior["pi_global"]), 4),
            "pi_success": round(float(prior["pi_success"]), 4),
            "neighbors": prior["neighbors"],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha0": float(self.alpha0),
            "beta0": float(self.beta0),
            "c_stable": float(self.c_stable),
            "max_events": int(self.max_events),
            "categorical": self.categorical.to_dict(),
            "events": list(self.events[-int(self.max_events) :]),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HSTGState":
        raw = dict(raw or {})
        return cls(
            alpha0=float(raw.get("alpha0", DEFAULT_ALPHA0)),
            beta0=float(raw.get("beta0", DEFAULT_BETA0)),
            c_stable=float(raw.get("c_stable", DEFAULT_C_STABLE)),
            categorical=CategoricalBayesState.from_dict(raw.get("categorical") or {}),
            events=[dict(item) for item in (raw.get("events") or [])],
            max_events=int(raw.get("max_events", MAX_EVENTS)),
        )
