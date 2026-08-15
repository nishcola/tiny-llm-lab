import json
from pathlib import Path

from tiny_llm_lab.app.experiment_results import render_experiment_results


class FakePage:
    def __init__(self) -> None:
        self.tables: list[object] = []
        self.charts: list[object] = []
        self.captions: list[str] = []

    def title(self, value: str) -> None: pass
    def caption(self, value: str) -> None: self.captions.append(value)
    def info(self, value: str) -> None: self.captions.append(value)
    def error(self, value: str) -> None: self.captions.append(value)
    def selectbox(self, label: str, options: list[str], **kwargs: object) -> str: return options[0]
    def dataframe(self, value: object, **kwargs: object) -> None: self.tables.append(value)
    def line_chart(self, value: object, **kwargs: object) -> None: self.charts.append(value)
    def subheader(self, value: str) -> None: pass
    def text_area(self, label: str, value: str, **kwargs: object) -> None: pass


def test_results_view_renders_summary_curves_and_seed_samples(tmp_path: Path) -> None:
    results = {
        "version": 1,
        "summary": [{"study": "position", "condition": "learned", "runs": 2, "validation_loss_mean": 1.5, "validation_loss_range": [1.4, 1.6], "parameter_count": 10, "training_seconds_mean": 2.0}],
        "runs": [
            {"study": "position", "condition": "learned", "seed": 1337, "validation_loss": 1.4, "metrics": [{"step": 1, "training_loss": 2.0, "validation_loss": 1.4}], "samples": [{"prompt": "ROMEO:", "continuation": "hello"}]},
            {"study": "position", "condition": "learned", "seed": 2027, "validation_loss": 1.6, "metrics": [{"step": 1, "training_loss": 2.1, "validation_loss": 1.6}], "samples": [{"prompt": "ROMEO:", "continuation": "world"}]},
        ],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    page = FakePage()

    render_experiment_results(page, path)

    assert len(page.tables) == 1
    assert len(page.charts) == 1
    assert "two-seed" in " ".join(page.captions).lower()
