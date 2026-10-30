"""``manage.py`` is the only entry point; experiment tooling is a library."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_utils_modules_are_not_entry_points():
    offenders = [
        path.relative_to(ROOT)
        for path in (ROOT / "utils").rglob("*.py")
        if "__main__" in path.read_text()
    ]

    assert offenders == []


def test_scripts_directory_is_gone():
    assert not (ROOT / "scripts").exists()


def test_manage_exposes_report_and_no_sweep():
    text = (ROOT / "manage.py").read_text()

    assert "add_parser('report'" in text
    assert "add_parser('sweep'" not in text
    assert not (ROOT / "utils" / "experiment" / "sweep.py").exists()


def test_report_takes_exactly_one_target():
    text = (ROOT / "manage.py").read_text()

    assert "add_mutually_exclusive_group(required=True)" in text
    assert "source.add_argument('--run'" in text
    assert "source.add_argument('--sweep'" in text
