import json
from unittest.mock import patch

import pytest

from sapai_gym1.rl.measurement_window import MeasurementWindow


def test_true_wall_counter_and_overhead_accounting(tmp_path):
    window = MeasurementWindow(tmp_path/'window.json')
    with patch('sapai_gym1.rl.measurement_window.time.perf_counter_ns', side_effect=[1_000_000_000, 5_000_000_000]), \
         patch('sapai_gym1.rl.measurement_window.time.time_ns', side_effect=[10, 20]):
        window.begin(100,120,11)
        window.record(update=11,before=100,after=200,decisions=100,primitives=110,
                      collect=1.,learn=1.,logging=.2,checkpoint=.3,maintenance=.4,telemetry=.1)
        result=window.finish(200,230,inference=lambda start,end:{'start':start,'end':end})
    assert result['acceptedDecisions']==100
    assert result['expandedPrimitives']==110
    assert result['wallDecisionsPerSecond']==25
    assert result['timedLoopDecisionsPerSecond']==50
    assert result['overheadSeconds']==2
    assert result['unclassifiedOverheadSeconds']==pytest.approx(1.)
    assert result['inference']=={'start':1_000_000_000,'end':5_000_000_000}
    assert json.loads((tmp_path/'window.json').read_text())['acceptedCounterEnd']==200


def test_bad_accepted_counter_fails_closed(tmp_path):
    window=MeasurementWindow(tmp_path/'window.json')
    window.begin(10,10,1)
    with pytest.raises(ValueError,match='counter delta'):
        window.record(update=1,before=10,after=15,decisions=6,primitives=6,collect=1,learn=1,
                      logging=0,checkpoint=0,maintenance=0,telemetry=0)


def test_telemetry_filters_completed_events_to_window():
    import multiprocessing as mp
    import threading
    from sapai_gym1.rl.distributed import DynamicInferenceBatcher, SharedActorBuffers
    from test_end_freeze_policy import policies
    env,policy,_=policies()
    context=mp.get_context('spawn')
    shared=SharedActorBuffers.create(env.observations()[0],policy.config.memory_dim,context,telemetry=True)
    batcher=DynamicInferenceBatcher(policy,[shared],'cpu',threading.RLock(),context.Event(),telemetry=True)
    batcher.response_samples.extend([(0.,2.,2000.),(2.,3.,1000.),(3.,6.,3000.)])
    batcher.group_samples.extend([(0.,0,10),(3.,0,4),(4.,7,2),(6.,7,8)])
    result=batcher.telemetry_snapshot(1_000_000_000,5_000_000_000)
    assert result['retainedResponseSamples']==1
    assert result['actorEnqueueToConsumeMilliseconds']['p95']==1000
    assert result['completedInferenceItemsByPolicyId']=={'0':4,'7':2}
