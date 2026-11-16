"""CPU collation and immutable, whole-match splits for standalone forecasting.

This module does not modify policy observations or a trained policy adapter.
Unknown fields and incomplete matches are reported, not silently imputed.
"""
from dataclasses import dataclass
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from .opponent_context_data import PublicOpponentContext, SCHEMA, validate_row

PROGRESS = ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0))
DATASET_SCHEMA = 'complete-public-forecast-splits-v1-20260906'
BOARD_KEYS = ('opponent_last', 'opponent_previous', 'our_last')
PET_FIELDS = {'species', 'perk', 'attack', 'health', 'level', 'experience',
              'temporaryAttack', 'temporaryHealth'}


@dataclass(frozen=True)
class PublicContextBatch:
    species: torch.Tensor
    perks: torch.Tensor
    numbers: torch.Tensor
    visible: torch.Tensor
    globals: torch.Tensor

    def to(self, device, *, non_blocking=False):
        # Exactly five batch transfers, never device allocations in pet loops.
        return PublicContextBatch(*(getattr(self, key).to(device, non_blocking=non_blocking)
                                    for key in self.__dataclass_fields__))

    def __len__(self):
        return self.species.shape[0]

    def history_only(self):
        """Ablate added causal features without changing encoder parameters.

        Keeps the two existing opponent boards and their missingness. This is
        a retrained same-encoder control, not exact production-head parity.
        """
        species, perks, numbers, visible, globals = (
            getattr(self, k).clone() for k in self.__dataclass_fields__)
        species[:, 10:] = 0; perks[:, 10:] = 0; numbers[:, 10:] = 0
        visible[:, 10:] = False
        globals[:, :5] = 0; globals[:, 7] = 1
        return PublicContextBatch(species, perks, numbers, visible, globals)


def validate_pet(pet, species_ids=None, perk_ids=None):
    if set(pet) != PET_FIELDS:
        raise ValueError('Unknown/missing public pet fields')
    if (type(pet['attack']) is not int or not 0 <= pet['attack'] <= 50
            or type(pet['health']) is not int or not 1 <= pet['health'] <= 50):
        raise ValueError('Invalid displayed pre-battle stats')
    if (type(pet['level']) is not int or type(pet['experience']) is not int
            or (pet['level'], pet['experience']) not in PROGRESS):
        raise ValueError('Invalid level/experience')
    if any(type(pet[k]) is not bool for k in ('temporaryAttack', 'temporaryHealth')):
        raise ValueError('Temporary amounts are private; only boolean flags are allowed')
    if pet['species'] == 'none':
        raise ValueError('Empty slots must be null, not occupied none pets')
    if species_ids is not None and pet['species'] not in species_ids:
        raise ValueError('Unknown species in production vocabulary')
    if perk_ids is not None and pet['perk'] not in perk_ids:
        raise ValueError('Unknown perk in production vocabulary')


def collate_contexts(contexts, species_ids, perk_ids):
    """Return owned CPU tensors. All Python work precedes any device transfer."""
    if not contexts:
        raise ValueError('An empty context batch is not a training batch')
    species = np.zeros((len(contexts), 15), dtype=np.int64)
    perks = np.zeros_like(species)
    numbers = np.zeros((len(contexts), 15, 7), dtype=np.float32)
    visible = np.zeros((len(contexts), 15), dtype=np.bool_)
    global_values = np.zeros((len(contexts), 8), dtype=np.float32)
    expected = set(PublicOpponentContext.__dataclass_fields__) | {'missing_history'}
    for b, context in enumerate(contexts):
        if set(context) != expected or context['pack'] != 'Turtle-P1-vs-P1':
            raise ValueError('Only the declared causal public context is allowed')
        missing = [context[key] is None for key in BOARD_KEYS]
        if (list(context['missing_history']) != missing
                or any(type(x) is not bool for x in context['missing_history'])):
            raise ValueError('Missing-history flags disagree with public boards')
        for field in ('turn', 'our_lives', 'opponent_lives', 'our_wins', 'opponent_wins'):
            if type(context[field]) is not int or context[field] < (1 if field == 'turn' else 0):
                raise ValueError('Invalid public turn/lives/wins')
        global_values[b] = [context['turn']/25, context['our_lives']/6,
                            context['opponent_lives']/6, context['our_wins']/25,
                            context['opponent_wins']/25, *missing]
        for role, key in enumerate(BOARD_KEYS):
            board = context[key]
            if board is None:
                continue
            if len(board) != 5:
                raise ValueError('Expected five public slots')
            visible[b, role*5:role*5+5] = True
            for position, pet in enumerate(board):
                if pet is None:
                    continue
                validate_pet(pet, species_ids, perk_ids)
                i = role*5+position
                species[b, i] = species_ids[pet['species']]
                perks[b, i] = perk_ids[pet['perk']]
                numbers[b, i] = [pet['attack']/50, pet['health']/50, pet['level']/3,
                                 pet['experience']/3, pet['temporaryAttack'],
                                 pet['temporaryHealth'], 1.]
    return PublicContextBatch(*(torch.from_numpy(x) for x in
                               (species, perks, numbers, visible, global_values)))


def collate_targets(boards, species_ids, perk_ids):
    labels = np.zeros((len(boards), 5, 6), dtype=np.int64)
    for b, board in enumerate(boards):
        if len(board) != 5:
            raise ValueError('Expected five target slots')
        for j, pet in enumerate(board):
            if pet is None:
                continue
            validate_pet(pet, species_ids, perk_ids)
            labels[b, j] = [species_ids[pet['species']], PROGRESS.index((pet['level'], pet['experience'])),
                            perk_ids[pet['perk']], pet['attack'], pet['health']-1,
                            int(pet['temporaryAttack'])+2*int(pet['temporaryHealth'])]
    return torch.from_numpy(labels)


def _match_error(rows):
    """Do not infer missing turns or use future duration to select individual rows."""
    seen = {}
    for row in rows:
        validate_row(row)
        context = row['context']
        missing = [context[key] is None for key in BOARD_KEYS]
        if (list(context['missing_history']) != missing
                or any(type(x) is not bool for x in context['missing_history'])):
            raise ValueError('Missing-history flags disagree with public boards')
        for board in (row['target'], *(context[key] for key in BOARD_KEYS)):
            if board is not None:
                for pet in board:
                    if pet is not None:
                        validate_pet(pet)
        meta = row['metadata']
        key = (meta['turn'], meta['side'])
        if meta['side'] not in (0, 1) or key in seen:
            raise ValueError('Duplicate/invalid perspective')
        seen[key] = row
    last = max(t for t, side in seen)
    if set(seen) != {(t, s) for t in range(1, last+1) for s in (0, 1)}:
        raise ValueError('Missing turn or perspective; incomplete match')
    if len({tuple(r['metadata']['checkpoint_populations']) for r in rows}) != 1:
        raise ValueError('Checkpoint populations changed within match')
    if len({r['metadata']['simulator_revision'] for r in rows}) != 1:
        raise ValueError('Mixed simulator revisions within match')
    canonical = lambda value: json.dumps(value, sort_keys=True)
    for (turn, side), row in seen.items():
        meta, context = row['metadata'], row['context']
        if len(meta['checkpoint_populations']) != 2 or meta['opponent_population'] != meta['checkpoint_populations'][1-side]:
            raise ValueError('Wrong opponent-population ownership')
        if type(meta['terminal_after_reveal']) is not bool or meta['terminal_after_reveal'] != (turn == last):
            raise ValueError('Missing or premature terminal reveal')
        expected = (seen[turn-1, side]['target'] if turn > 1 else None,
                    seen[turn-2, side]['target'] if turn > 2 else None,
                    seen[turn-1, 1-side]['target'] if turn > 1 else None)
        if any(canonical(context[k]) != canonical(v) for k, v in zip(BOARD_KEYS, expected)):
            raise ValueError('History does not match prior public reveals')


def freeze_match_splits(examples, *, heldout_populations=(), validation_fraction=.15,
                        test_fraction=.15, salt='forecast-quality-v1', require_nonempty=True):
    """Freeze all perspectives/turns together; report every rejected whole match."""
    if min(validation_fraction, test_fraction) < 0 or validation_fraction+test_fraction >= 1:
        raise ValueError('Invalid split fractions')
    groups = defaultdict(list)
    if len({row['metadata']['simulator_revision'] for row in examples}) > 1:
        raise ValueError('A forecasting-quality dataset must use one simulator revision')
    for row in examples:
        meta = row['metadata']
        groups[str(meta['collection_id']), str(meta['match_id'])].append(row)
    splits = {key: [] for key in ('train', 'validation', 'test')}
    assignments, rejected = [], []
    heldout = set(heldout_populations)
    for key, rows in sorted(groups.items()):
        try:
            _match_error(rows)
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({'match': list(key), 'rows': len(rows), 'reason': str(error)})
            continue
        populations = set(rows[0]['metadata']['checkpoint_populations'])
        score = int(hashlib.sha256((salt+repr(key)).encode()).hexdigest()[:16], 16)/2**64
        split = ('test' if populations & heldout or score < test_fraction else
                 'validation' if score < test_fraction+validation_fraction else 'train')
        splits[split].extend(sorted(rows, key=lambda row: (row['metadata']['turn'], row['metadata']['side'])))
        assignments.append({'match': list(key), 'split': split, 'populations': sorted(populations)})
    metadata = {'schema': DATASET_SCHEMA, 'rowSchema': SCHEMA, 'salt': salt,
                'heldoutPopulations': sorted(heldout), 'validationFraction': validation_fraction,
                'testFraction': test_fraction, 'inputRows': len(examples),
                'acceptedRows': sum(map(len, splits.values())), 'rejectedMatches': rejected,
                'rejectedRows': sum(x['rows'] for x in rejected),
                'assignments': assignments, 'futureSurvivalFiltering': False,
                'terminalRows': {s: sum(r['metadata']['terminal_after_reveal'] for r in rows) for s, rows in splits.items()},
                'rowsBySplit': {s: len(rows) for s, rows in splits.items()},
                'matchesBySplit': dict(Counter(a['split'] for a in assignments))}
    if require_nonempty and any(not rows for rows in splits.values()):
        raise ValueError('A quality experiment requires nonempty train, validation and test matches')
    return {'metadata': metadata, **splits}
