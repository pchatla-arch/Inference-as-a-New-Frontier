#!/usr/bin/env python3
"""Proposed accounting equations, not a performance simulator or RL implementation.

Counts logical tokens; it does not convert tokens into FLOPs, joules, or GPU time.
The command-line example is synthetic and makes no empirical improvement claim.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import argparse
import json
import math
from typing import Iterable, Mapping


def _nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be numeric, not {type(value).__name__}')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class RoundDemand:
    candidates: int
    evaluations_per_candidate: float
    generated_candidate_tokens: float
    generated_evaluation_tokens: float
    generator_prompt_tokens: float
    evaluator_context_tokens: float  # Excludes the candidate, counted separately.

    def __post_init__(self) -> None:
        if isinstance(self.candidates, bool) or not isinstance(self.candidates, int):
            raise TypeError('candidates must be an integer')
        for name, value in asdict(self).items():
            _nonnegative(name, value)


@dataclass(frozen=True)
class TokenDemand:
    logical_decode_tokens: float
    logical_prompt_tokens: float


def token_demand(rounds: Iterable[RoundDemand], *, extra_decode: float=0,
                 extra_prompt: float=0) -> TokenDemand:
    """Approximate demand under a common tokenization convention.

    Use actual trace sums for correlated lengths or variable judge multiplicity.
    Extras are repairs/meta-evaluation NOT already included in the round inputs.
    Prefix caching changes executed prefill, not these logical prompt counts.
    """
    _nonnegative('extra_decode', extra_decode)
    _nonnegative('extra_prompt', extra_prompt)
    dec = float(extra_decode); prompt = float(extra_prompt)
    for r in rounds:
        if not isinstance(r, RoundDemand):
            raise TypeError('rounds must contain RoundDemand objects')
        dec += r.candidates * (r.generated_candidate_tokens +
               r.evaluations_per_candidate * r.generated_evaluation_tokens)
        prompt += r.candidates * (r.generator_prompt_tokens +
                  r.evaluations_per_candidate *
                  (r.evaluator_context_tokens + r.generated_candidate_tokens))
    if not all(math.isfinite(x) for x in (dec,prompt)):
        raise OverflowError('Demand overflowed; reduce or partition the inputs')
    return TokenDemand(dec,prompt)


def bytes_per_verified_improvement(boundary_bytes: Mapping[str,float],
                                   accepted_improvements: int) -> dict[str,float | None]:
    """Bytes per distinct protocol-verified improvement, by measured boundary.

    None denotes undefined when zero improvements were accepted. Boundary totals
    may include multiple crossings of the same physical data; do not infer energy.
    """
    if isinstance(accepted_improvements,bool) or not isinstance(accepted_improvements,int):
        raise TypeError('accepted_improvements must be an integer')
    _nonnegative('accepted_improvements', accepted_improvements)
    for h, value in boundary_bytes.items():_nonnegative(h,value)
    return {h:(value/accepted_improvements if accepted_improvements else None)
            for h,value in boundary_bytes.items()}


def weighted_movement_cost(boundary_bytes: Mapping[str,float],
                           cost_per_byte: Mapping[str,float],
                           accepted_improvements: int) -> float | None:
    """Weighted cost per improvement; NOT a byte-count metric.

    Coefficients must all use one unit, e.g. J/byte. Missing coefficients are errors.
    """
    bytes_per_verified_improvement(boundary_bytes,accepted_improvements)
    if set(boundary_bytes) != set(cost_per_byte):
        raise ValueError('Exactly one coefficient is required for each boundary')
    for h,v in cost_per_byte.items():_nonnegative(h,v)
    if accepted_improvements==0:return None
    result=sum(boundary_bytes[h]*cost_per_byte[h] for h in boundary_bytes)/accepted_improvements
    if not math.isfinite(result):raise OverflowError('Weighted cost overflowed')
    return result


def quality_gain_per_joule(quality_change: float, energy_joules: Mapping[str,float]) -> float:
    """Signed quality change / total campaign energy from disjoint categories.

    The caller must use a fixed external quality protocol and include unsuccessful
    work. This helper cannot establish evaluator validity or energy measurement.
    """
    if isinstance(quality_change,bool) or not isinstance(quality_change,(int,float)):
        raise TypeError('quality_change must be numeric')
    if not math.isfinite(quality_change):raise ValueError('quality_change must be finite')
    required={'generation','evaluation','tools','training','synchronization'}
    if set(energy_joules)!=required:raise ValueError('Provide the five disjoint campaign energy categories')
    for k,v in energy_joules.items():_nonnegative(k,v)
    total=sum(energy_joules.values())
    if not math.isfinite(total) or total<=0:raise ValueError('Total energy must be finite and positive')
    return quality_change/total


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example',action='store_true',help='Run a synthetic token-accounting sanity check')
    args=parser.parse_args()
    if not args.example:parser.error('Use --example; library functions accept measured input records')
    r=RoundDemand(32,2,512,64,256,128)
    result=token_demand([r]*3)
    print(json.dumps({'evidence_class':'synthetic accounting example; not measured',
                      'rounds':3,'round_input':asdict(r),**asdict(result)},indent=2))

if __name__=='__main__':main()
