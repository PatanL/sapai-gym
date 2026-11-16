"""Construct public search inputs from either observed history or an opening prior."""
from collections import Counter

from .rollout_search import PublicSearchState


def public_state_from_reservoir(core, side, reservoir, *, horizon=5, limit=32, opening_seed=None):
    if side not in (0, 1):
        raise ValueError('side must be zero or one')
    latest = core.previous_opponent_state if side == 0 else core.previous_agent_state
    if latest is None:
        if opening_seed is None:
            raise ValueError('A no-history search requires a declared opening-prior seed')
        matches = reservoir.opening_prior(seed=opening_seed, horizon=horizon, limit=limit)
        state = PublicSearchState.from_core(core, side, opening_prior=[match.teams for match in matches])
        kind = 'opening_population'
    else:
        if opening_seed is not None:
            raise ValueError('Opening priors must not replace observed history')
        own_previous = core.previous_agent_state if side == 0 else core.previous_opponent_state
        older = core.previous_previous_opponent_state if side == 0 else core.previous_previous_agent_state
        own_older = core.previous_previous_agent_state if side == 0 else core.previous_previous_opponent_state
        matches = reservoir.match(latest, observed_enemy=own_previous, older_observed=older,
                                  older_observed_enemy=own_older, current_turn=core.turn,
                                  horizon=horizon, limit=limit)
        if not matches:
            raise ValueError('No public opponent trajectories match this state')
        state = PublicSearchState.from_core(core, side)
        kind = 'matched_public_history'
    templates = state.opponent_trajectories if kind == 'opening_population' else [match.teams for match in matches]
    state.set_opponent_trajectories(templates,
                                    [match.trajectory_id for match in matches])
    state.set_opponent_transition_model(reservoir, candidate_limit=limit)
    return state, {
        'kind': kind, 'openingSeed': opening_seed,
        'candidateIds': [match.trajectory_id for match in matches],
        'sourceCounts': dict(sorted(Counter(str(match.source_index) for match in matches).items())),
    }
