#!/usr/bin/env python3
"""Replay complete recorded actions WITHOUT neural inference, verify, then label.

Never guesses missing temporary flags from displayed stats. The verified exact
simulator replay supplies them. Mismatches reject the whole match with a reason.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction
from sapai_gym1.rl.game_trace import player_snapshot, team_snapshot
from sapai_gym1.rl.opponent_context_data import SCHEMA, OneStepRevealCollector, validate_row


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint_path(path, project):
    if path.startswith('/workspace/'):
        return project/path.removeprefix('/workspace/')
    return Path(path)


def reconstruct(p0, p1, populations, match_id):
    meta = p0['metadata']
    for key in ('seed', 'candidateCheckpoint', 'opponentCheckpoint', 'simulatorRevision'):
        if p1['metadata'][key] != meta[key]:
            raise ValueError('Perspectives do not identify the same match')
    if meta['simulatorRevision'] != SIMULATOR_REVISION:
        raise ValueError('Different simulator revision; no invented state migration')
    for doc in (p0, p1):
        if doc['summary']['result'] not in (0, 1, 2):
            raise ValueError('Missing complete match result')
    frames = [[f for f in doc['frames'] if f.get('action')] for doc in (p0, p1)]
    battle_frames = [{f['turn']: f for f in doc['frames'] if f['phase'] == 'battle'} for doc in (p0, p1)]
    env = ParallelSelfPlayEnv(); env.reset(seed=meta['seed']); indices = [0, 0]
    collector = OneStepRevealCollector(simulator_revision=SIMULATOR_REVISION, collection_id='verified-trace-replay-v1')
    collector.begin_turn(env.core, match_id, populations=populations)
    while not env.core.is_done():
        actions = []
        for side, active in enumerate(env.active_players()):
            if not active:
                actions.append(None); continue
            if indices[side] >= len(frames[side]):
                raise ValueError('Recorded actions end before terminal match')
            frame = frames[side][indices[side]]
            if frame['turn'] != env.core.turn:
                raise ValueError('Wrong recorded turn')
            player = env.core.agent if side == 0 else env.core.opponent
            if player_snapshot(player) != frame['before']['player']:
                raise ValueError(f'Exact pre-action state mismatch turn={env.core.turn} side={side} index={indices[side]}')
            a = frame['action']; actions.append(AutoregressiveAction(ActionType[a['type']], a['source'], a['target']))
            indices[side] += 1
        turn = env.core.turn
        transition = env.step(*actions)
        if not transition.battle_resolved:
            continue
        submitted = (env.core.previous_agent_state, env.core.previous_opponent_state)
        for side in (0, 1):
            recorded = battle_frames[side].get(turn)
            if not recorded or recorded['battle']['result'] != transition.battle_result:
                raise ValueError('Battle outcome mismatch')
            if (recorded['battle']['viewerTeam'] != team_snapshot(submitted[side])
                    or recorded['battle']['opponentTeam'] != team_snapshot(submitted[1-side])):
                raise ValueError('Submitted board mismatch')
        collector.revealed(match_id=match_id, turn=turn, submitted_boards=submitted, terminal=transition.terminated)
        if not transition.terminated:
            collector.begin_turn(env.core, match_id, populations=populations)
    if indices != [len(x) for x in frames]:
        raise ValueError('Trailing actions after terminal result')
    for row in collector.examples:
        validate_row(row)
    return collector.examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument('--project', type=Path, default=Path('/workspace'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new immutable output')
    groups = defaultdict(dict); rejected = []; inputs = []
    for path in args.traces:
        doc = json.loads(path.read_text()); meta = doc['metadata']
        key = (meta['candidateCheckpoint'], meta['opponentCheckpoint'], meta['seed'], meta['simulatorRevision'])
        side = meta['perspective']
        if side in groups[key]:
            raise ValueError('Duplicate trace perspective')
        groups[key][side] = doc
        inputs.append({'path': str(path), 'sha256': sha(path)})
    rows = []; count = 0
    for key, pair in sorted(groups.items()):
        match_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()
        try:
            if set(pair) != {'p0', 'p1'}:
                raise ValueError('Missing complementary perspective')
            populations = tuple(sha(checkpoint_path(path, args.project)) for path in key[:2])
            added = reconstruct(pair['p0'], pair['p1'], populations, match_id)
        except (ValueError, RuntimeError, KeyError) as error:
            rejected.append({'match': match_id, 'reason': str(error)}); continue
        rows.extend(added); count += 1
    payload = {'schema': SCHEMA, 'metadata': {'generatorSHA256': sha(Path(__file__)), 'inputs': inputs,
        'games': count, 'rejectedMatches': rejected, 'simulator_revision': SIMULATOR_REVISION,
        'selectionCaveat': 'Archived viewer games may be selected/curated; use as auxiliary data, not a representative final test population',
        'neuralEvaluations': 0, 'trainingActivated': False}, 'train': rows, 'heldout': []}
    with args.output.open('x') as output:
        json.dump(payload, output, indent=2); output.write('\n')
    print(json.dumps({'games': count, 'rows': len(rows), 'rejected': rejected, 'neuralEvaluations': 0}))


if __name__ == '__main__':
    main()
