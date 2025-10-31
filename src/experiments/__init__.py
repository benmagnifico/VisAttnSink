"""
Experimental modules for VAR assumption validation.
"""

from .grounding_utils import RefCOCOLoader, BoundingBoxMapper
from .harmful_recycling import HarmfulRecyclingExperiment
from .logit_ranking import LogitRankingExperiment

__all__ = [
    'RefCOCOLoader',
    'BoundingBoxMapper',
    'HarmfulRecyclingExperiment',
    'LogitRankingExperiment',
]
