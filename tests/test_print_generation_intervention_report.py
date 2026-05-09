from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "print_generation_intervention_report.py"
    spec = importlib.util.spec_from_file_location(
        "print_generation_intervention_report",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_results_groups_by_prompt_and_alpha() -> None:
    script = load_script()

    grouped = script.group_results(
        [
            {"prompt": "Review:", "alpha": 0.0, "completion": " okay"},
            {"prompt": "Review:", "alpha": 2.0, "completion": " great"},
            {"prompt": "Other:", "alpha": 0.0, "completion": " neutral"},
        ],
    )

    assert grouped["Review:"][0.0] == [" okay"]
    assert grouped["Review:"][2.0] == [" great"]
    assert grouped["Other:"][0.0] == [" neutral"]
