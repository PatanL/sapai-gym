from pathlib import Path

import pytest

from sapai_gym1.rl.league import LeagueManager, LeagueMember


def test_interrupted_save_preserves_previous_league(tmp_path, monkeypatch):
    path = tmp_path / 'league.json'
    league = LeagueManager()
    league.add(LeagueMember('parent', 'parent.pt'))
    league.save(path)
    original = path.read_bytes()
    league.add(LeagueMember('next', 'next.pt'))
    replace = Path.replace

    def interrupted(source, target):
        if source == path.with_suffix('.json.tmp'):
            raise OSError('simulated interruption before publication')
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', interrupted)
        with pytest.raises(OSError, match='interruption'):
            league.save(path)
    assert path.read_bytes() == original
    assert set(LeagueManager.load(path).members) == {'parent'}
    league.save(path)
    assert set(LeagueManager.load(path).members) == {'parent', 'next'}
    assert not path.with_suffix('.json.tmp').exists()
