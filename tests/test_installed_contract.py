"""Reporting without bundled source configuration has an actionable boundary."""
from pathlib import Path
import pytest


def test_report_missing_checkout_config_explains_explicit_config(tmp_path, monkeypatch):
    from signed_study import report
    monkeypatch.setattr(report,'__file__',str(tmp_path/'site-packages'/'signed_study'/'report.py'))
    with pytest.raises(ValueError, match='config_path'):
        report.report_config(tmp_path/'empty-results')


def test_report_explicit_config_works_without_checkout(tmp_path, monkeypatch):
    from signed_study import report
    config=Path(__file__).resolve().parents[1]/'config/study.json'
    monkeypatch.setattr(report,'__file__',str(tmp_path/'site-packages'/'signed_study'/'report.py'))
    assert report.report_config(tmp_path/'empty-results',config_path=config).paths['results']==tmp_path/'empty-results'
