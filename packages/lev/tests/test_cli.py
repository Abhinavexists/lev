from dataclasses import replace

from lev import cli
from lev.train import loop
from lev.train.config import PRESETS


def test_measuring_a_plan_does_not_change_the_next_plans_defaults(monkeypatch, capsys):
    preset = replace(PRESETS["smoke"])
    monkeypatch.setitem(PRESETS, "smoke", preset)
    default_tokens = preset.avg_tokens_per_example
    measured = {"n": 10, "mean": 321, "median": 300, "p95": 400, "max": 450}
    monkeypatch.setattr(cli, "_measure_tokens", lambda *args: measured)

    cli.main(["plan", "--preset", "smoke", "--data", "unused"])
    assert "321 tok" in capsys.readouterr().out

    cli.main(["plan", "--preset", "smoke"])
    assert f"{default_tokens} tok" in capsys.readouterr().out
    assert preset.avg_tokens_per_example == default_tokens


def test_training_output_override_does_not_change_the_next_runs_defaults(monkeypatch, tmp_path):
    preset = replace(PRESETS["smoke"])
    monkeypatch.setitem(PRESETS, "smoke", preset)
    default_output = preset.output_dir
    override = str(tmp_path / "custom-output")
    outputs = []

    def run_training(config, data_dir, **kwargs):
        outputs.append(config.output_dir)
        return {
            "steps": 1,
            "first_loss": 1.0,
            "final_loss": 0.5,
            "output_dir": config.output_dir,
        }

    monkeypatch.setattr(loop, "run_training", run_training)
    cli.main(["train", "--preset", "smoke", "--output-dir", override])
    cli.main(["train", "--preset", "smoke"])

    assert outputs == [override, default_output]
    assert preset.output_dir == default_output
