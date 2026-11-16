"""Read-only battle snapshots for the isolated, single-thread replay generator.

No second combat implementation or extra RNG draws. Instrumentation is scoped
to this context, never enabled in actors/training. Phase snapshots supplement
the engine's existing ability-log emission points; a phase may group effects.
"""
from contextlib import contextmanager
import importlib

from .game_trace import team_snapshot


@contextmanager
def capture_battle_playback():
    engine = importlib.import_module('sapai.battle')
    environment = importlib.import_module('sapai_gym1.rl.selfplay_env')
    original_battle = environment.Battle
    original_phase, original_append = engine.battle_phase, engine.append_phase_list
    records, active = [], []

    def snapshot(battle, label, kind, detail=None):
        boards = [team_snapshot(battle.t0), team_snapshot(battle.t1)]
        events = battle.replay_events
        if events and boards == events[-1]['teams'] and kind == 'phase':
            return
        events.append({'label': label, 'kind': kind, 'teams': boards, 'detail': detail})

    class RecordedBattle(original_battle):
        def battle(self):
            self.replay_events = []
            active.append(self)
            snapshot(self, 'Submitted boards', 'initial')
            try:
                result = super().battle()
                snapshot(self, 'Battle complete', 'complete')
                records.append({'schema': 'battle-playback-v1', 'result': int(result),
                                'granularity': 'engine-trigger-log-and-phase-boundaries',
                                'events': self.replay_events})
                return result
            finally:
                active.pop()

    def phase(battle, name, teams, priority, phase_dict):
        result = original_phase(battle, name, teams, priority, phase_dict)
        if active and active[-1] is battle:
            snapshot(battle, name.removeprefix('phase_').replace('_', ' ').title(), 'phase')
        return result

    def append(phase_list, pet, team_index, pet_index, activated, targets, possible):
        result = original_append(phase_list, pet, team_index, pet_index, activated, targets, possible)
        if activated and active:
            label = pet.name.removeprefix('pet-').replace('-', ' ').title()
            trigger = str(pet.ability.get('trigger', 'Ability'))
            snapshot(active[-1], f'{label}: {trigger}', 'trigger',
                     {'seat': int(team_index), 'slot': int(pet_index), 'pet': pet.name})
        return result

    environment.Battle, engine.battle_phase, engine.append_phase_list = RecordedBattle, phase, append
    try:
        yield records
    finally:
        environment.Battle = original_battle
        engine.battle_phase, engine.append_phase_list = original_phase, original_append
