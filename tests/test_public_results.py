"""The public checker must reject damaged or internally inconsistent exports."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / 'scripts' / 'verify_public_results.py'


@pytest.fixture
def public_copy(tmp_path):
    docs = tmp_path / 'docs'
    docs.mkdir()
    manifest = json.loads((ROOT / 'docs/public-results.json').read_text())
    for relative in manifest['artifacts']:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (docs / 'public-results.json').write_text(json.dumps(manifest))
    return tmp_path


def check(root):
    return subprocess.run(
        [sys.executable, '-I', str(CHECKER), '--root', str(root)],
        capture_output=True, text=True, check=False,
    )


def rewrite_csv(root, name, mutate):
    path = root / 'docs' / name
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    mutate(rows)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = root / 'docs/public-results.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts'][f'docs/{name}']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_clean_aggregates_verify_without_installed_package(public_copy):
    result = check(public_copy)
    assert result.returncode == 0, result.stderr
    assert '13 methods' in result.stdout
    assert '12 paired comparisons' in result.stdout
    assert 'informative negative' in result.stdout


def test_changed_csv_bytes_fail_hash_check(public_copy):
    path = public_copy / 'docs/native-confirmation-metrics.csv'
    path.write_bytes(path.read_bytes() + b'\n')
    result = check(public_copy)
    assert result.returncode != 0
    assert 'SHA256 mismatch' in result.stderr
    assert path.name in result.stderr


@pytest.mark.parametrize('name,field', [
    ('native-confirmation-metrics.csv', 'method'),
    ('native-confirmation-comparisons.csv', 'comparator'),
    ('native-timing-summary.csv', 'method'),
])
def test_duplicate_identity_fails_even_with_updated_hash(public_copy, name, field):
    rewrite_csv(public_copy, name, lambda rows: rows[1].update({field: rows[0][field]}))
    result = check(public_copy)
    assert result.returncode != 0
    assert 'duplicate' in result.stderr.lower()


def test_primary_direction_cannot_be_flipped_in_manifest(public_copy):
    path = public_copy / 'docs/public-results.json'
    manifest = json.loads(path.read_text())
    manifest['primary']['difference_definition'] = 'comparator minus SignedMax'
    path.write_text(json.dumps(manifest))
    result = check(public_copy)
    assert result.returncode != 0
    assert 'difference_definition' in result.stderr


def test_nonfinite_auc_fails_even_with_updated_hash(public_copy):
    rewrite_csv(public_copy, 'native-confirmation-metrics.csv',
                lambda rows: rows[0].update(auc='nan'))
    result = check(public_copy)
    assert result.returncode != 0
    assert 'finite' in result.stderr


def test_missing_confirmation_observation_fails_even_with_updated_hash(public_copy):
    rewrite_csv(public_copy, 'native-confirmation-metrics.csv',
                lambda rows: rows[0].update(n_completed='1775'))
    result = check(public_copy)
    assert result.returncode != 0
    assert '1776' in result.stderr


def test_changed_comparison_mean_fails_even_with_updated_hash(public_copy):
    rewrite_csv(public_copy, 'native-confirmation-comparisons.csv',
                lambda rows: rows[0].update(sample_mean_low='-0.028'))
    result = check(public_copy)
    assert result.returncode != 0
    assert 'mean' in result.stderr.lower()
