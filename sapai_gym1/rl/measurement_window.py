"""Opt-in benchmark accounting around complete post-warmup updates."""
import json
from pathlib import Path
import time


class MeasurementWindow:
    def __init__(self, output):
        self.output = Path(output)
        self.start_ns = self.end_ns = None
        self.rows = []

    def begin(self, counter, primitive_counter, update):
        self.start_ns = time.perf_counter_ns()
        self.start_utc_ns = time.time_ns()
        self.counter_start = int(counter)
        self.primitives_start = int(primitive_counter)
        self.first_update = int(update)

    def record(self, *, update, before, after, decisions, primitives, collect, learn,
               logging, checkpoint, maintenance, telemetry):
        if self.start_ns is None:
            return
        if int(after)-int(before) != int(decisions):
            raise ValueError('Actual learner counter delta disagrees with accepted decisions')
        self.rows.append(dict(update=int(update), counterBefore=int(before), counterAfter=int(after),
            decisions=int(decisions), primitives=int(primitives), collectSeconds=collect,
            learnPublishSeconds=learn, loggingSeconds=logging, checkpointSeconds=checkpoint,
            maintenanceSeconds=maintenance, telemetrySeconds=telemetry))

    def finish(self, counter, primitive_counter, *, inference=None, memory=None):
        if self.start_ns is None:
            return
        self.end_ns = time.perf_counter_ns()
        self.end_utc_ns = time.time_ns()
        if callable(inference):
            inference = inference(self.start_ns, self.end_ns)
        accepted = int(counter) - self.counter_start
        if accepted != sum(row['decisions'] for row in self.rows):
            raise ValueError('Measurement window decision count mismatch')
        wall = (self.end_ns-self.start_ns)/1e9
        parts = {key: sum(row[key] for row in self.rows) for key in (
            'collectSeconds','learnPublishSeconds','loggingSeconds','checkpointSeconds',
            'maintenanceSeconds','telemetrySeconds')}
        loop = parts['collectSeconds'] + parts['learnPublishSeconds']
        payload = dict(version='complete-update-wall-window-v2', startMonotonicNs=self.start_ns,
            endMonotonicNs=self.end_ns, startUtcNs=self.start_utc_ns, endUtcNs=self.end_utc_ns,
            acceptedCounterStart=self.counter_start, acceptedCounterEnd=int(counter),
            acceptedDecisions=accepted, primitiveCounterStart=self.primitives_start,
            primitiveCounterEnd=int(primitive_counter), expandedPrimitives=int(primitive_counter)-self.primitives_start,
            measuredWallSeconds=wall, timedLoopSeconds=loop, wallDecisionsPerSecond=accepted/wall,
            timedLoopDecisionsPerSecond=accepted/loop, overheadSeconds=wall-loop,
            unclassifiedOverheadSeconds=wall-sum(parts.values()), parts=parts, rows=self.rows,
            inference=inference, memory=memory,
            scope='Whole post-warmup updates including logging, checkpoints and resident maintenance; excludes startup/shutdown and report serialization. Telemetry events filtered to this window.')
        temp = self.output.with_suffix('.tmp')
        temp.write_text(json.dumps(payload, indent=2)+'\n')
        temp.replace(self.output)
        return payload
