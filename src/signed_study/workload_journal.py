"""Versioned append-only workload evidence and strict, outcome-free recovery.

Version 2 retains starts before evaluation, one terminal record, measured journal
duration, per-entry timings, finalized per-user overhead/resources and a final
materialization digest. A missing final event or torn final append is incomplete,
even when all visible attempts terminated. Recovery never evaluates or retries.
Historical version 1 journals permit identical/enriched repeated terminal rows;
they cannot reconstruct timing/resource fields absent from those old events.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .protocol import canonical_sha256
from .metrics import METRIC_KEYS


_MAX_RECORD_BYTES = 16 * 1024 * 1024
_IDENTITY_KEYS = ('evaluation_index', 'user_id', 'configuration_id', 'family',
                  'parameters', 'kind', 'chunk_size')
_TERMINAL = ('completed', 'failed', 'interrupted')
_STATE_KEYS = ('status', 'chunk_invariance', 'repair_decisions', 'fixed_seconds',
               'optional_seconds', 'memory_bound', 'prototype_cap_user_fraction',
               'resources', 'stop_reason')
_EVENT_FIELDS = {
    'workload': {'inputs', 'identity'},
    'attempt_started': {'attempt'},
    'attempt_terminal': {'attempt'},
    'attempt_timing': {'evaluation_index', 'write_seconds'},
    'measurement': {'evaluation_index', 'timed_evaluation_index', 'write_seconds', 'measurement'},
    'user_completed': {'user_id', 'resource', 'measurements', 'state'},
    'state': {'state'},
    'finished': {'measurements_sha256'},
}


class JournalWriteError(OSError):
    """A journal write failed; do not append behind a possibly torn record."""


def initial_measurements():
    return dict(status='completed', attempts=[], actual_evaluations=0,
                user_measurements=[], resource_trace=[], chunk_invariance=True,
                repair_decisions=[], fixed_seconds=0., optional_seconds=0., memory_bound=False)


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _integer(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def _duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _verify_event(event):
    if (not isinstance(event, dict) or type(event.get('schema_version')) is not int
            or event['schema_version'] not in (1, 2)
            or not _digest(event.get('input_sha256'))
            or event.get('output_sha256') != canonical_sha256(
                {k: v for k, v in event.items() if k != 'output_sha256'})):
        raise ValueError('Workload journal envelope/schema/digest discrepancy')


def _verify_identity(identity, inputs):
    if (not isinstance(inputs, dict) or set(inputs) !=
            {'episodes', 'user_ids', 'entries', 'numerical_checks', 'timing_only'}
            or not _digest(inputs.get('episodes'))
            or not isinstance(inputs.get('user_ids'), list)
            or any(type(u) is not int for u in inputs['user_ids'])
            or not isinstance(inputs.get('entries'), list)
            or any(not isinstance(e, str) or not e for e in inputs['entries'])
            or type(inputs.get('numerical_checks')) is not bool
            or type(inputs.get('timing_only')) is not bool
            or inputs['timing_only'] and inputs['numerical_checks']
            or not isinstance(identity, dict) or identity.get('inputs') != inputs
            or not isinstance(identity.get('config'), dict)):
        raise ValueError('Workload journal input identity discrepancy')
    users = identity.get('users', [])
    entries = identity.get('entries', [])
    if (not isinstance(users, list) or not isinstance(entries, list) or not entries
            or any(not isinstance(r, dict) for r in [*users, *entries])
            or [r.get('user_id') for r in users] != inputs.get('user_ids')
            or [r.get('id') for r in entries] != inputs.get('entries')
            or len(set(inputs['user_ids'])) != len(users)
            or len(set(inputs['entries'])) != len(entries)
            or not _integer(identity.get('score_chunk_size'), positive=True)
            or not isinstance(identity.get('score_chunk_sizes_check'), list)
            or any(not _integer(c, positive=True) for c in identity['score_chunk_sizes_check'])):
        raise ValueError('Invalid workload journal schedule identity')
    for row in users:
        if (type(row.get('user_id')) is not int
                or any(not _integer(row.get(k)) for k in
                       ('positive_count', 'negative_count', 'candidate_count'))
                or not _digest(row.get('candidate_ids_sha256'))):
            raise ValueError('Invalid workload journal user identity')
    for row in entries:
        if (not isinstance(row.get('id'), str) or not row['id']
                or not isinstance(row.get('family'), str) or not row['family']
                or not isinstance(row.get('parameters'), dict)):
            raise ValueError('Invalid workload journal configuration identity')
    numerics = identity['config'].get('numerics')
    if numerics is not None and (not isinstance(numerics, dict)
            or numerics.get('score_chunk_size') != identity['score_chunk_size']
            or numerics.get('score_chunk_sizes_check') != identity['score_chunk_sizes_check']):
        raise ValueError('Workload journal chunk policy differs from execution config')


class WorkloadJournal:
    """Exclusive fresh writer; an existing journal is evidence, never a retry cue."""

    def __init__(self, path, inputs, identity):
        self.path = Path(path)
        _verify_identity(identity, inputs)
        self.input_sha256 = canonical_sha256(inputs)
        self.workload_sha256 = canonical_sha256(identity)
        self.sequence = 0
        self.failed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also refuses old/torn journals; callers choose a new
        # admitted workload identity rather than silently rerunning a prior fit.
        with self.path.open('xb') as handle:
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.append('workload', inputs=inputs, identity=identity)

    def append(self, event, **payload):
        if self.failed:
            raise JournalWriteError('Workload journal is unavailable after a failed append')
        value = dict(schema_version=2, input_sha256=self.input_sha256,
                     workload_sha256=self.workload_sha256, sequence=self.sequence,
                     event=event, **payload)
        value['output_sha256'] = canonical_sha256(value)
        encoded = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ValueError('Workload journal record exceeds the 16 MiB bound')
        try:
            with self.path.open('ab') as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            self.failed = True
            raise JournalWriteError(f'Workload journal append failed: {error}') from error
        self.sequence += 1


def _validate_attempt(row):
    if (not isinstance(row, dict)
            or not _integer(row.get('evaluation_index'))
            or type(row.get('user_id')) is not int
            or not isinstance(row.get('configuration_id'), str)
            or not isinstance(row.get('family'), str)
            or not isinstance(row.get('parameters'), dict)
            or row.get('kind') not in ('fit_and_score', 'score_repeat')
            or not _integer(row.get('chunk_size'), positive=True)
            or row.get('status') not in ('started', *_TERMINAL)
            or not isinstance(row.get('diagnostics'), dict)):
        raise ValueError('Invalid workload journal attempt identity/status')
    for key in ('fit_seconds', 'score_seconds', 'write_seconds', 'elapsed_seconds', 'diagnostic_seconds'):
        if key in row and not _duration(row[key]):
            raise ValueError('Invalid workload journal attempt duration')
    if row['status'] == 'completed':
        if not _duration(row.get('score_seconds')):
            raise ValueError('Completed workload journal attempt lacks score timing')
        if row['kind'] == 'fit_and_score' and not _duration(row.get('fit_seconds')):
            raise ValueError('Completed workload journal fit lacks fit timing')
        if row['kind'] == 'fit_and_score' or 'scores' in row:
            scores, ids = row.get('scores'), row.get('candidate_ids')
            if (not isinstance(scores, list) or not isinstance(ids, list)
                    or len(scores) != len(ids) or any(type(i) is not int for i in ids)
                    or any(type(s) not in (int, float) or not math.isfinite(s) for s in scores)):
                raise ValueError('Invalid completed workload journal score/candidate schema')


def _same_identity(a, b):
    return canonical_sha256({k:a.get(k) for k in _IDENTITY_KEYS}) == canonical_sha256(
        {k:b.get(k) for k in _IDENTITY_KEYS})


def _enrich(previous, current):
    if any(key in current and current[key] != value for key, value in previous.items()):
        raise ValueError('Conflicting duplicate workload journal terminal')
    return {**previous, **current}


def replay_workload_journal(path, *, expected_input_sha256=None, expected_config=None):
    """Validate and reconstruct saved evidence without opening any outcome arrays.

    No newline means an uncommitted final append, including syntactically valid
    JSON. Internal corruption, conflicting terminals and mixed inputs fail closed.
    ``complete`` means a version-2 final digest is present and intact; successful
    stage/ledger ownership is a separate consumer gate.
    """
    result = initial_measurements()
    attempts = {}
    measured = {}
    user_resources = []
    header = None
    version = digest = workload_digest = None
    finalized = False
    torn = b''
    sequence = 0
    seen_queries = set()
    query_indices = {}
    timed = set()
    finished_users = []
    finished_set = set()
    users = entries = {}
    open_index = None
    canonical_completed = set()

    def set_timing(index, seconds):
        if (index not in attempts or attempts[index]['status'] == 'started'
                or index in timed or not _duration(seconds)):
            raise ValueError('Invalid workload journal terminal timing')
        attempts[index]['write_seconds'] = seconds
        timed.add(index)

    def set_measurement(index, value):
        row = attempts.get(index)
        if (row is None or row['kind'] != 'fit_and_score' or row['status'] != 'completed'
                or not isinstance(value, dict)
                or any(value.get(k) != row[k] for k in ('user_id', 'configuration_id', 'family', 'parameters'))
                or any(not _duration(value.get(k)) for k in
                       ('fit_seconds', 'score_seconds', 'write_seconds', 'overhead_seconds'))
                or any(value.get(k) != row.get(k) for k in ('fit_seconds', 'score_seconds', 'write_seconds'))):
            raise ValueError('Invalid workload journal measurement identity/timing')
        user = users[row['user_id']]
        if (any(value.get(k) != user[k] for k in ('positive_count', 'negative_count', 'candidate_count'))
                or value.get('support_count') != user['positive_count'] + user['negative_count']
                or value.get('chunk_size') != identity['score_chunk_size']
                or not isinstance(value.get('score_measurements'), dict)
                or str(identity['score_chunk_size']) not in value['score_measurements']):
            raise ValueError('Workload journal measurement size/chunk discrepancy')
        previous = measured.get(index)
        if previous is not None:
            fixed = set(previous) - {'overhead_seconds', 'score_measurements'}
            if (any(previous[k] != value.get(k) for k in fixed)
                    or value['overhead_seconds'] < previous['overhead_seconds']):
                raise ValueError('Conflicting workload journal measurement update')
        for chunk, timing in value['score_measurements'].items():
            if (not chunk.isdigit() or int(chunk) not in
                    {identity['score_chunk_size'], *identity['score_chunk_sizes_check']}
                    or any(not _duration(timing.get(k)) for k in
                           ('score_seconds', 'write_seconds', 'overhead_seconds'))
                    or type(timing.get('numerical_accepted')) is not bool):
                raise ValueError('Invalid workload journal measured chunk')
            chunk_index = query_indices.get((row['user_id'], row['configuration_id'], int(chunk)))
            chunk_row = attempts.get(chunk_index)
            if (chunk_row is None or chunk_index not in timed
                    or timing['score_seconds'] != chunk_row.get('score_seconds', 0.)
                    or timing['write_seconds'] != chunk_row['write_seconds']
                    or timing['numerical_accepted'] != (chunk_row['status'] == 'completed')):
                raise ValueError('Workload journal timing differs from its actual terminal attempt')
            if previous is not None and chunk in previous['score_measurements']:
                old = previous['score_measurements'][chunk]
                if (any(timing.get(k) != v for k, v in old.items() if k != 'overhead_seconds')
                        or timing['overhead_seconds'] < old['overhead_seconds']):
                    raise ValueError('Conflicting workload journal chunk timing update')
        if previous is not None and not previous['score_measurements'].keys() <= value['score_measurements'].keys():
            raise ValueError('Workload journal measurement removed a chunk')
        measured[index] = deepcopy(value)

    def set_state(value):
        if (not isinstance(value, dict) or set(value) - set(_STATE_KEYS)
                or value.get('status', result['status']) not in ('completed', 'repair_required', 'interrupted')
                or type(value.get('chunk_invariance', result['chunk_invariance'])) is not bool
                or not isinstance(value.get('repair_decisions', result['repair_decisions']), list)):
            raise ValueError('Invalid workload journal state')
        if not result['chunk_invariance'] and value.get('chunk_invariance') is True:
            raise ValueError('Workload journal cannot reverse a numerical failure')
        if (not all(item in value.get('repair_decisions', result['repair_decisions'])
                    for item in result['repair_decisions'])
                or result['status'] == 'repair_required' and value.get('status') == 'completed'
                or any(k in value and not _duration(value[k]) for k in ('fixed_seconds', 'optional_seconds'))
                or 'prototype_cap_user_fraction' in value and
                (not _duration(value['prototype_cap_user_fraction']) or value['prototype_cap_user_fraction'] > 1)):
            raise ValueError('Workload journal state reverses failures or has invalid durations')
        result.update(deepcopy(value))
    with Path(path).open('rb') as handle:
        while True:
            raw = handle.readline(_MAX_RECORD_BYTES + 1)
            if not raw:
                break
            if len(raw) > _MAX_RECORD_BYTES:
                raise ValueError('Workload journal record exceeds the 16 MiB bound')
            if not raw.endswith(b'\n'):
                torn = raw
                break
            try:
                event = json.loads(raw)
                _verify_event(event)
            except (ValueError, UnicodeError, TypeError) as error:
                raise ValueError(f'Invalid complete workload journal record {sequence}') from error
            if version is None:
                version, digest = event['schema_version'], event['input_sha256']
                if expected_input_sha256 is not None and digest != expected_input_sha256:
                    raise ValueError('Workload journal does not match expected inputs')
            if event['schema_version'] != version or event['input_sha256'] != digest:
                raise ValueError('Mixed workload journal schemas/inputs')
            if version == 1:
                if set(event) != {'schema_version', 'input_sha256', 'attempt', 'output_sha256'}:
                    raise ValueError('Invalid historical workload journal envelope')
                row = event['attempt']
                _validate_attempt(row)
                index = row['evaluation_index']
                previous = attempts.get(index)
                if row['status'] == 'started':
                    if previous is not None or index != len(attempts):
                        raise ValueError('Duplicate/out-of-order historical workload start')
                elif previous is None or not _same_identity(previous, row):
                    raise ValueError('Historical workload terminal lacks matching start')
                elif previous['status'] != 'started':
                    row = _enrich(previous, row)
                attempts[index] = row
                sequence += 1
                continue
            if event.get('sequence') != sequence or type(event.get('sequence')) is not int:
                raise ValueError('Workload journal sequence discrepancy')
            if finalized:
                raise ValueError('Workload journal has records after finalization')
            kind = event.get('event')
            expected_fields = _EVENT_FIELDS.get(kind)
            envelope = {'schema_version', 'input_sha256', 'workload_sha256', 'sequence', 'event', 'output_sha256'}
            if expected_fields is None or set(event) != envelope | expected_fields:
                raise ValueError('Invalid complete workload journal record fields')
            if sequence == 0:
                if kind != 'workload':
                    raise ValueError('Workload journal lacks initial identity')
                header = event
                _verify_identity(event.get('identity'), event.get('inputs'))
                workload_digest = canonical_sha256(event['identity'])
                identity = header['identity']
                users = {r['user_id']: r for r in identity['users']}
                entries = {r['id']: r for r in identity['entries']}
                if canonical_sha256(event['inputs']) != digest:
                    raise ValueError('Workload journal input digest discrepancy')
                if expected_config is not None and identity['config'] != expected_config:
                    raise ValueError('Workload journal does not match verified execution config')
            if event.get('workload_sha256') != workload_digest:
                raise ValueError('Mixed workload journal identities')
            identity = header['identity']
            if kind == 'workload':
                if sequence != 0:
                    raise ValueError('Duplicate workload journal identity')
            elif kind in ('attempt_started', 'attempt_terminal'):
                row = event.get('attempt')
                _validate_attempt(row)
                index = row['evaluation_index']
                previous = attempts.get(index)
                user, entry = users.get(row['user_id']), entries.get(row['configuration_id'])
                if (user is None or entry is None or row['family'] != entry['family']
                        or canonical_sha256(row['parameters']) != canonical_sha256(entry['parameters'])):
                    raise ValueError('Workload journal attempt is outside bound schedule')
                query = (row['user_id'], row['configuration_id'], row['chunk_size'])
                canonical = identity['score_chunk_size']
                if (row['kind'] == 'fit_and_score' and row['chunk_size'] != canonical
                        or row['kind'] == 'score_repeat' and
                        (not header['inputs']['numerical_checks'] or row['chunk_size'] == canonical
                         or row['chunk_size'] not in identity['score_chunk_sizes_check'])):
                    raise ValueError('Workload journal attempt chunk/kind discrepancy')
                if kind == 'attempt_started':
                    if (row['status'] != 'started' or previous is not None or index != len(attempts)
                            or query in seen_queries or open_index is not None
                            or row['user_id'] in finished_set):
                        raise ValueError('Invalid workload journal start transition')
                    if (row['kind'] == 'score_repeat'
                            and (row['user_id'], row['configuration_id']) not in canonical_completed):
                        raise ValueError('Repeated score lacks successful canonical attempt')
                    seen_queries.add(query)
                    query_indices[query] = index
                    open_index = index
                elif (row['status'] not in _TERMINAL or previous is None
                        or previous['status'] != 'started' or not _same_identity(previous, row)):
                    raise ValueError('Invalid/conflicting workload journal terminal transition')
                else:
                    open_index = None
                    if row['kind'] == 'fit_and_score' and row['status'] == 'completed':
                        canonical_completed.add((row['user_id'], row['configuration_id']))
                if kind == 'attempt_terminal' and (row['status'] == 'completed' or 'metrics' in row):
                    metrics = row.get('metrics')
                    if 'metrics' not in row:
                        raise ValueError('Completed workload journal attempt lacks metrics boundary')
                    if header['inputs']['timing_only']:
                        if metrics is not None:
                            raise ValueError('Timing-only workload cannot claim outcome metrics')
                    elif (not isinstance(metrics, dict) or set(metrics) != set(METRIC_KEYS)
                            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                                   for v in metrics.values())):
                        raise ValueError('Invalid completed workload journal metrics schema')
                    if row['kind'] == 'score_repeat':
                        difference = row.get('metric_difference')
                        if (not isinstance(difference, dict) or set(difference) != set(METRIC_KEYS)
                                or any(type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1
                                       for v in difference.values())
                                or type(row.get('accepted')) is not bool
                                or row['accepted'] != (row['status'] == 'completed')):
                            raise ValueError('Invalid workload journal numerical-repeat evidence')
                if row['status'] == 'completed' and 'candidate_ids' in row:
                    if (len(row['candidate_ids']) != user['candidate_count']
                            or canonical_sha256({'item_ids':row['candidate_ids']}) != user['candidate_ids_sha256']):
                        raise ValueError('Workload journal candidates differ from bound input')
                attempts[index] = deepcopy(row)
            elif kind == 'attempt_timing':
                set_timing(event.get('evaluation_index'), event.get('write_seconds'))
            elif kind == 'measurement':
                set_timing(event.get('timed_evaluation_index'), event.get('write_seconds'))
                set_measurement(event.get('evaluation_index'), event.get('measurement'))
            elif kind == 'user_completed':
                resource = event.get('resource')
                uid = event.get('user_id')
                if (not isinstance(resource, dict) or resource.get('user_id') != uid
                        or len(finished_users) >= len(identity['users'])
                        or uid != identity['users'][len(finished_users)]['user_id']
                        or open_index is not None
                        or any((uid, entry['id'], identity['score_chunk_size']) not in seen_queries
                               for entry in identity['entries'])):
                    raise ValueError('Invalid workload journal user completion')
                finished_users.append(uid)
                finished_set.add(uid)
                user_resources.append(resource)
                values = event.get('measurements')
                if not isinstance(values, list):
                    raise ValueError('Workload user completion lacks final timing coverage')
                indices = [v.get('evaluation_index') for v in values if isinstance(v, dict)]
                expected_indices = {query_indices[(uid, entry['id'], identity['score_chunk_size'])]
                                    for entry in identity['entries']
                                    if (uid, entry['id']) in canonical_completed}
                if len(indices) != len(values) or len(set(indices)) != len(indices) or set(indices) != expected_indices:
                    raise ValueError('Workload user completion has duplicate or missing final measurements')
                if header['inputs']['numerical_checks'] and any(
                        (uid, entry['id'], chunk) not in seen_queries
                        for entry in identity['entries'] if (uid, entry['id']) in canonical_completed
                        for chunk in identity['score_chunk_sizes_check']):
                    raise ValueError('Workload user completion lacks scheduled numerical repeats')
                for value in values:
                    index = value.get('evaluation_index')
                    if index not in attempts or attempts[index]['user_id'] != uid:
                        raise ValueError('Workload user completion has foreign timing identity')
                    set_measurement(index, value.get('measurement'))
                set_state(event.get('state'))
            elif kind == 'state':
                set_state(event.get('state'))
            elif kind == 'finished':
                result.update(attempts=[attempts[i] for i in range(len(attempts))],
                              actual_evaluations=len(attempts),
                              user_measurements=[measured[i] for i in sorted(measured)],
                              resource_trace=user_resources)
                if (finished_users != header['inputs']['user_ids']
                        or any(r['status'] == 'started' for r in attempts.values())
                        or any(r['status'] == 'completed' and r['kind'] == 'fit_and_score' and i not in measured
                               for i, r in attempts.items())
                        or timed != set(attempts)
                        or event.get('measurements_sha256') != canonical_sha256(result)):
                    raise ValueError('Workload journal final materialization discrepancy')
                finalized = True
            else:
                raise ValueError('Unknown complete workload journal record')
            sequence += 1
    result.update(attempts=[attempts[i] for i in range(len(attempts))], actual_evaluations=len(attempts),
                  user_measurements=[measured[i] for i in sorted(measured)], resource_trace=user_resources)
    complete = finalized and not torn
    if not complete:
        result['status'] = 'interrupted'
    return dict(schema_version=version, input_sha256=digest, workload_sha256=workload_digest,
                complete=complete, measurements=result, torn_tail_bytes=len(torn),
                torn_tail_sha256=hashlib.sha256(torn).hexdigest() if torn else None,
                legacy_fields_unavailable=['user_measurements', 'resource_trace', 'overhead'] if version == 1 else [])
