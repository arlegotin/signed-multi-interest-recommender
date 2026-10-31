#!/usr/bin/env python3
"""Fixed-input old/prepared forecast mechanism profile; no study data or fits."""
import argparse
import json
from pathlib import Path
import platform
import time
import tracemalloc

from signed_study.calibration import _fit_cost, _score_units
from signed_study.costs import PreparedCosts
from signed_study.data import Episode, History
from signed_study.models import model_configuration
from signed_study.stage_evidence import file_reference


def legacy(episodes,entries,rows,counter):
    total=0.
    for entry in entries:
        priced=[r for r in rows if r['family']==entry.family]
        for episode in episodes:
            counter[0]+=len(priced)
            h=episode.history
            rate=max(r['score_seconds']/_score_units(entry.family,r['positive_count'],r['negative_count'],
                r['candidate_count'],r['K']) for r in priced)
            total+=(_fit_cost(priced,len(h.positive_ids)+len(h.negative_ids))
                +_score_units(entry.family,len(h.positive_ids),len(h.negative_ids),len(episode.candidate_ids),2)*rate
                +max(r['write_seconds']+r['overhead_seconds'] for r in priced))
    return total


def profile(repeats):
    results=[]
    entries=[model_configuration(f,{'K':2,'lambda':2}) for f in
             ('SignedMax','MeanNegative','AllExemplarMargin')]
    for n in (16,32,64):
        episodes=[Episode(History(i,0,'development',tuple(range(10+i%20)),tuple(range(100,105+i%4))),
                          tuple(range(200,232+i))) for i in range(n)]
        rows=[dict(family=family,support_count=15+i%25,positive_count=10+i%25,negative_count=5,
                   candidate_count=32+i,fit_seconds=.0001*(1+i%4),score_seconds=.0002,
                   write_seconds=.00003,overhead_seconds=.00002,K=2,chunk_size=512)
              for i in range(n) for family in
              ('SignedMax','MeanNegative','AllExemplarMargin','PrototypeRatio','ExemplarRatio','LinearLogistic','RBFSVM')
              for _ in range(3)]
        observations=[]
        for repeat in range(repeats):
            for method in (('legacy','prepared') if repeat%2==0 else ('prepared','legacy')):
                tracemalloc.start();start=time.perf_counter_ns();counter=[0]
                if method=='legacy':
                    cost=legacy(episodes,entries,rows,counter)
                    preparation=None
                else:
                    before=time.perf_counter_ns()
                    prepared=PreparedCosts(rows,chunk_size=512)
                    for entry in entries:prepared.envelope(entry)
                    preparation=(time.perf_counter_ns()-before)/1e9
                    cost=prepared.total(episodes,entries)
                    counter[0]=sum(1 for row in rows if row['family'] in {e.family for e in entries})
                elapsed=(time.perf_counter_ns()-start)/1e9
                _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                observations.append(dict(method=method,repeat=repeat,total_seconds=elapsed,
                    preparation_seconds=preparation,fit_bin_row_visits=counter[0],peak_tracemalloc_bytes=peak,
                    forecast_seconds=cost))
        assert len({r['forecast_seconds'] for r in observations})==1
        results.append(dict(episodes=n,timing_rows=len(rows),entries=len(entries),observations=observations))
    return dict(schema_version=1,python=platform.python_version(),source_sha256={str(p):file_reference(p)['sha256']
        for p in (Path(__file__),Path('src/signed_study/costs.py'),Path('src/signed_study/calibration.py'))},
        boundary='in-memory preparation plus forecasting under tracemalloc; no fit, score, filesystem or ledger work',
        results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('.cache/integrity/forecast-profile.json'))
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    result=profile(args.repeats)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(args.output)
