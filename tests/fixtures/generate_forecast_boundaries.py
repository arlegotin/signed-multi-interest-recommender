"""Generate runtime-specific goldens from the inspected, unmodified forecaster.

From the checkout, with the selected Python and installed test dependencies:
    PYTHONPATH=src:tests python tests/fixtures/generate_forecast_boundaries.py

No acquisition, fitting or historical result writes occur. The original 3.12
fixture remains unchanged. Python 3.11 uses a different builtin float sum.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from conftest import study_settings
from test_calibration import cohort, measured
from signed_study.calibration import forecast_scale
from signed_study.protocol import canonical_sha256, load_config


def main():
    directory = Path(__file__).resolve().parent
    template = json.loads((directory / 'forecast-boundaries.json').read_text())
    source = subprocess.check_output(['git', 'show',
        template['baseline_commit'] + ':src/signed_study/calibration.py'])
    assert hashlib.sha256(source).hexdigest() == template['baseline_source_sha256']
    namespace = {'__package__':'signed_study', '__name__':'signed_study._forecast_baseline'}
    exec(compile(source, '<verified baseline calibration>', 'exec'), namespace)
    assert canonical_sha256(measured()) == template['measurement_sha256']
    version = '.'.join(map(str,sys.version_info[:2]))
    if version not in ('3.11','3.12'):
        raise ValueError('Only the supported Python 3.11 and 3.12 runtimes are accepted')
    with tempfile.TemporaryDirectory(prefix='forecast-fixture-') as temporary:
        path = Path(temporary) / 'study.json'
        path.write_text(json.dumps(study_settings.__wrapped__()))
        config = load_config(path)
        if version == '3.11':
            state = template['cases'][0]['state']
            ledger = SimpleNamespace(config=config,snapshot=lambda:state)
            normal = namespace['forecast_scale'](measured(),cohort(),ledger)
            threshold = config.resources['safety_multiplier'] * normal['seconds']['G3']
            limits = {'G3_below':math.nextafter(threshold,-math.inf),
                      'G3_at':threshold,'G3_above':math.nextafter(threshold,math.inf)}
            for case in template['cases']:
                if case['name'] in limits:
                    case['state']['stage_limits']['G3'] = limits[case['name']]
        for case in template['cases']:
            ledger = SimpleNamespace(config=config,snapshot=lambda:case['state'])
            expected = namespace['forecast_scale'](measured(),cohort(),ledger)
            actual = forecast_scale(measured(),cohort(),ledger)
            assert actual == expected, case['name']
            if case['name'] == 'G3_below':
                assert expected['scale'] == [256,1776]
            elif case['name'] in ('G3_at','G3_above'):
                assert expected['scale'] == [418,1776]
            case.update(forecast_sha256=canonical_sha256(expected),
                        scale=expected['scale'],status=expected['status'])
    if version == '3.12':
        assert template == json.loads((directory / 'forecast-boundaries.json').read_text())
        print('Original Python 3.12 goldens verified; no files changed')
        return
    template['python'] = version
    target = directory / ('forecast-boundaries.python' + version + '.json')
    target.write_text(json.dumps(template,sort_keys=True,indent=2) + '\n')
    print('Verified baseline/new equality; wrote',target)


if __name__ == '__main__':
    main()
