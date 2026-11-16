import copy

import pytest
import torch

from sapai_gym1.rl.opponent_context_data import PublicOpponentContext, SCHEMA
from sapai_gym1.rl.opponent_distribution import AutoregressiveOpponentBoard, OpponentDistributionConfig
from sapai_gym1.rl.opponent_forecast_data import collate_contexts, collate_targets, freeze_match_splits


PET = {'species': 'pet-fish', 'perk': 'none', 'attack': 3, 'health': 4,
       'level': 1, 'experience': 1, 'temporaryAttack': False, 'temporaryHealth': True}
BOARD = [PET, None, None, None, None]


def contexts():
    return [PublicOpponentContext(None, None, None, 1, 6, 6, 0, 0).features(),
            PublicOpponentContext(BOARD, None, BOARD, 2, 5, 6, 0, 1).features()]


def complete_match(name, populations=('a', 'b')):
    rows = []
    for turn in (1, 2):
        for side in (0, 1):
            rows.append({'schema': SCHEMA, 'context': contexts()[turn-1], 'target': copy.deepcopy(BOARD),
                         'metadata': {'collection_id': 'collection', 'match_id': str(name), 'turn': turn,
                                      'side': side, 'checkpoint_populations': populations,
                                      'opponent_population': populations[1-side], 'simulator_revision': 'test',
                                      'visibility': 'submitted-pre-battle-reveal', 'terminal_after_reveal': turn == 2}})
    return rows


def test_prepacked_cpu_batches_are_exactly_equivalent_with_gradients():
    torch.manual_seed(7)
    model = AutoregressiveOpponentBoard(OpponentDistributionConfig(('none', 'pet-fish'), ('none',), width=32))
    batch = collate_contexts(contexts(), model.species_ids, model.perk_ids)
    targets = collate_targets([BOARD, [None]*5], model.species_ids, model.perk_ids)
    assert all(getattr(batch, key).device.type == 'cpu' for key in batch.__dataclass_fields__)
    assert targets.device.type == 'cpu'
    first = model.log_prob(contexts(), [BOARD, [None]*5])
    first.sum().backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    second = model.log_prob(batch, targets)
    second.sum().backward()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    for n, p in model.named_parameters():
        if n in gradients:
            torch.testing.assert_close(gradients[n], p.grad, atol=0, rtol=0)


def test_production_vocabulary_width128_budget_and_real_fields():
    from sapai.data import data
    from sapai_gym1.rl.entity_observation import EntityObservationEncoder
    encoder = EntityObservationEncoder()
    species = ('none', *sorted(k for k in data['pets'] if k != 'pet-none'))
    model = AutoregressiveOpponentBoard(OpponentDistributionConfig(species, tuple(encoder.status_to_id)))
    assert model.config.width == 128
    count = sum(p.numel() for p in model.parameters())
    assert count < 1_000_000
    with torch.no_grad():
        assert torch.isfinite(model.log_prob(contexts(), [BOARD, [None]*5])).all()


def test_history_only_control_cannot_read_added_features_and_does_not_mutate_batch():
    base = contexts()[1]
    changed = copy.deepcopy(base)
    changed.update(turn=10, our_lives=1, opponent_wins=5, our_last=[None]*5)
    first = collate_contexts([base], {'none': 0, 'pet-fish': 1}, {'none': 0})
    second = collate_contexts([changed], {'none': 0, 'pet-fish': 1}, {'none': 0})
    for key in first.__dataclass_fields__:
        torch.testing.assert_close(getattr(first.history_only(), key), getattr(second.history_only(), key), atol=0, rtol=0)
    assert first.visible[:, 10:].all()
    assert first.globals[0, 0] != second.globals[0, 0]


def test_marginal_control_has_no_previous_slot_or_species_field_conditioning():
    from sapai_gym1.rl.opponent_distribution import MarginalOpponentBoard
    model = MarginalOpponentBoard(OpponentDistributionConfig(('none', 'pet-fish'), ('none',), width=32))
    encoded = model.encode_context(contexts())
    targets = model.target_tensor([BOARD, BOARD]); altered = targets.clone()
    altered[:, 0, 3] = 20
    calls = []
    handle = model.species_head.register_forward_pre_hook(lambda _, args: calls.append(args[0].detach().clone()))
    model.decode(encoded, targets); first = calls[1].clone(); calls.clear()
    model.decode(encoded, altered); second = calls[1].clone(); handle.remove()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    loss = -model.log_prob(contexts(), [BOARD, BOARD]).mean(); loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('mutation', ['missing', 'private', 'unknown', 'progress', 'temporary_amount'])
def test_bad_public_or_target_fields_fail_closed(mutation):
    context = copy.deepcopy(contexts()[1]); board = copy.deepcopy(BOARD)
    if mutation == 'missing': context['missing_history'][0] = True
    if mutation == 'private': context['shop'] = []
    if mutation == 'unknown': board[0]['species'] = 'pet-not-real'
    if mutation == 'progress': board[0]['experience'] = 9
    if mutation == 'temporary_amount': board[0]['temporaryHealth'] = 3
    with pytest.raises(ValueError):
        collate_contexts([context], {'none': 0, 'pet-fish': 1}, {'none': 0})
        collate_targets([board], {'none': 0, 'pet-fish': 1}, {'none': 0})


def test_complete_splits_are_reproducible_match_grouped_and_population_held_out():
    rows = sum([complete_match(i) for i in range(100)], []) + complete_match('unseen', ('a', 'unseen'))
    first = freeze_match_splits(rows, heldout_populations=('unseen',))
    second = freeze_match_splits(list(reversed(rows)), heldout_populations=('unseen',))
    assert first == second
    assignments = {a['match'][1]: a['split'] for a in first['metadata']['assignments']}
    assert assignments['unseen'] == 'test'
    assert len(assignments) == 101
    assert first['metadata']['rejectedRows'] == 0
    assert sum(first['metadata']['terminalRows'].values()) == 202
    assert set(first['metadata']['matchesBySplit']) == {'train', 'validation', 'test'}


@pytest.mark.parametrize('corruption', ['missing_side', 'missing_terminal', 'future_history', 'duplicate'])
def test_incomplete_or_contaminated_matches_are_reported_not_imputed(corruption):
    rows = complete_match('bad')
    if corruption == 'missing_side': rows.pop()
    if corruption == 'missing_terminal': rows[-1]['metadata']['terminal_after_reveal'] = False
    if corruption == 'future_history': rows[0]['context']['opponent_last'] = copy.deepcopy(BOARD)
    if corruption == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    result = freeze_match_splits(rows, require_nonempty=False)
    assert result['metadata']['acceptedRows'] == 0
    assert result['metadata']['rejectedRows'] == len(rows)
    assert len(result['metadata']['rejectedMatches']) == 1
