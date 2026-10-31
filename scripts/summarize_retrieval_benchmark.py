#!/usr/bin/env python3
"""Recompute all paired timing statistics without loading geometry or histories."""
import argparse
import json
from pathlib import Path
from signed_study.retrieval_benchmark import summarize

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('observations',type=Path)
args=parser.parse_args()
saved=json.loads(args.observations.read_text())
config=saved['configuration']
schedule=[(row['workload'],row['ordinal']) for row in saved['workloads']]
expected={(workload,ordinal) for workload in config['workload_names'] for ordinal in range(config['users'])}
if len(schedule)!=len(expected) or set(schedule)!=expected:
    raise ValueError('Saved workload inventory differs from declared schedule')
summary=summarize(saved['observations'],build_seconds=saved['boundaries']['build_seconds'],
    expected_repeats=saved['configuration']['repeats'],expected_ordinals=range(saved['configuration']['users']),
    expected_methods=saved['configuration']['methods'],expected_workloads=config['workload_names'])
if summary!=saved['summary']:
    raise ValueError('Saved summary does not match raw observations')
print(json.dumps(summary,sort_keys=True,indent=2))
