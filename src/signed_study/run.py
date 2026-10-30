"""CLI-facing target inspection and ordered acquisition/calibration interfaces."""

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import platform
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np
from threadpoolctl import threadpool_info

from .budget import Ledger, atomic_write_json
from .protocol import StudyConfig, canonical_sha256
from .calibration import select_calibration_users, forecast_scale
from .metrics import select_family, select_comparator
from .models import endpoint_requests


def develop(config: StudyConfig, ledger: Ledger) -> dict:
    return run_stage('develop', config, ledger)


def run_stage(stage: str, config: StudyConfig, ledger: Ledger) -> dict:
    from .stages import run_stage as execute
    return execute(stage, config, ledger)


_PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'torch', 'sentence-transformers',
             'huggingface-hub', 'psutil', 'matplotlib', 'pytest')


def _command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
        return dict(command=argv, returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=argv, returncode=None, stdout='', stderr=str(error))


def _hardware_evidence():
    commands = {name: _command(command) for name, command in {
        'uname': ['uname', '-a'], 'architecture': ['uname', '-m'],
        'macos': ['sw_vers'], 'chip': ['sysctl', '-n', 'machdep.cpu.brand_string'],
        'ram': ['sysctl', '-n', 'hw.memsize'], 'logical_cpus': ['sysctl', '-n', 'hw.logicalcpu'],
        'translated': ['sysctl', '-n', 'sysctl.proc_translated'],
        'power_source': ['pmset', '-g', 'batt'], 'power_mode': ['pmset', '-g', 'custom'],
        'low_power': ['osascript', '-l', 'JavaScript', '-e',
                      'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.isLowPowerModeEnabled'],
        'disk': ['df', '-h', '.'],
    }.items()}

    def successful(name):
        result = commands[name]
        return result['stdout'] if result['returncode'] == 0 else None

    def integer(name):
        value = successful(name)
        return int(value) if value is not None and value.isdecimal() else None

    power_source = successful('power_source')
    low_power = successful('low_power')
    # proc_translated is absent on native Apple silicon, but a permission failure
    # is unknown, never evidence for native execution.
    translation = successful('translated')
    translated_result = commands['translated']
    native_absence = ('unknown oid' in translated_result['stderr'].lower() and
                      platform.machine() == 'arm64')
    translated = translation == '1' if translation in ('0', '1') else (False if native_absence else None)
    return dict(system=platform.system(), machine=platform.machine(), os=platform.platform(),
                macos=successful('macos'), chip=successful('chip'),
                installed_ram_bytes=integer('ram'), logical_cpus=integer('logical_cpus'),
                translated=translated, commands=commands,
                power=dict(ac=('AC Power' in power_source) if power_source else None,
                           low_power_mode=(low_power == 'true') if low_power in ('true', 'false') else None,
                           low_power_method='Foundation.NSProcessInfo.isLowPowerModeEnabled via JXA; raw pmset settings also retained'))


def _json_value(value):
    if hasattr(value, 'items'):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def inspect_environment(config: StudyConfig) -> dict:
    """Save measured environment or explicit blockers; never load models/outcomes.

    This G0 metadata operation is charged to the same persistent ledger as later
    stages. Sandbox-denied sysctl/pmset fields stay unknown with command stderr;
    rerun with authorized host read access to establish target evidence.
    """
    results = config.paths['results']
    ledger = Ledger(results / 'budget_ledger.jsonl', config)
    try:
        with ledger.work('G0', projected_seconds=15, evaluations=0) as attempt:
            hardware = _hardware_evidence()
            sample = ledger.checkpoint()
            blockers = []
            if hardware['system'] != 'Darwin' or hardware['chip'] != 'Apple M3 Max':
                blockers.append('Requested Apple M3 Max target is not verified; inspect hardware command evidence.')
            if hardware['machine'] != 'arm64' or hardware['translated'] is not False:
                blockers.append('Native arm64 execution is not verified.')
            if hardware['installed_ram_bytes'] is None or hardware['logical_cpus'] is None:
                blockers.append('Actual RAM/CPU counts could not be verified.')
            if hardware['power']['ac'] is not True:
                blockers.append('AC power is not verified; plug in before target workloads.')
            if hardware['power']['low_power_mode'] is not False:
                blockers.append('Low-power mode is active or not verified.')
            if sys.version_info[:2] not in ((3, 11), (3, 12)):
                blockers.append('Python 3.11/3.12 is required.')
            if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != (Path.cwd() / '.venv').resolve():
                blockers.append('The project-local .venv is required.')
            if sample['available_memory_bytes'] < config.resources['available_memory_headroom_bytes']:
                blockers.append('Less than 4 GiB estimated available memory headroom.')
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                np.show_config()
            packages = {}
            for name in _PACKAGES:
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
                    blockers.append(f'Required installed package not found: {name}')
            settings = {field.name: _json_value(getattr(config, field.name)) for field in fields(config)}
            bootstrap = results / 'bootstrap_environment.json'
            inputs = dict(config_sha256=canonical_sha256(settings),
                          bootstrap_environment_sha256=hashlib.sha256(bootstrap.read_bytes()).hexdigest() if bootstrap.exists() else None)
            evidence = dict(schema_version=1, inspected_at=datetime.now(timezone.utc).isoformat(),
                            hardware=hardware, resources=sample, python=dict(version=sys.version,
                            executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix),
                            packages=packages, numpy_blas=buffer.getvalue(), threadpools=threadpool_info(),
                            thread_policy=dict(blas_threads=config.timing['blas_threads'], inference_workers=1,
                            environment={name: os.environ.get(name) for name in
                                         ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}),
                            contention=dict(load_average=list(os.getloadavg()),
                            method='os.getloadavg; system-wide load; no attribution to other applications'),
                            measurement_method=dict(rss='psutil process + recursive live children; 100 ms interval; OS self maximum also recorded',
                            available_memory='psutil.virtual_memory().available',
                            disk='project lstat bytes including .venv/.cache/.git; full scan at boundaries, cached up to 5 seconds'),
                            target_compatible=not blockers, blockers=blockers, experimental_workload=False,
                            inputs=inputs)
            evidence['output_sha256'] = canonical_sha256(evidence)
            artifact = results / 'environment.json'
            atomic_write_json(artifact, evidence)
            ledger.finish(attempt, 'blocked' if blockers else 'completed', {},
                          {'environment': {'path': str(artifact),
                                           'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}})
        return evidence
    finally:
        ledger.close()
