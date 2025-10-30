# tui/regression_tab.py
import numpy as np
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll, Grid
from textual.widgets import Static, Input, Label
from textual.validation import Number, Function

from tui.helper import sanitize_id


class RegressionTab(Static):
    DEFAULT_CSS = """
    .section {
        margin: 1 0;
        padding: 1;
        border: solid $primary;
        height: auto;
    }
    #inputs {
        grid-size: 2;
        grid-gutter: 1;
        height: auto;
    }
    """

    def __init__(
        self,
        model_name: str,
        feature_names: list[str],
        coeffs: np.ndarray,
        r: float,
        r_squared: float,
        initial_values: dict[str, float] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.feature_names = feature_names
        self.coeffs = coeffs
        self.r = r
        self.r_squared = r_squared
        self.initial_values = initial_values or {}
        self.inputs = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            # Model statistics section
            with Vertical(classes="section") as stats_section:
                stats_section.border_title = f"📊 {self.model_name}"
                yield Label(f"Correlation (r): {self.r:.4f}")
                yield Label(f"Coefficient of Determination (R²): {self.r_squared:.4f}")

            # Input section
            with Vertical(classes="section") as input_section:
                input_section.border_title = "🩺 Input Features"
                with Grid(id="inputs"):
                    for feat in self.feature_names:
                        label = Label(f"{feat}:")
                        validator = (
                            Number(minimum=0)
                            if "Activities" not in feat
                            else Function(lambda x: x in ("0", "1"), "Enter 0 or 1")
                        )
                        safe_id = f"input-{sanitize_id(feat)}"
                        input_widget = Input(
                            placeholder=f"Enter {feat}",
                            validators=[validator],
                            id=safe_id,
                        )
                        self.inputs[feat] = input_widget
                        yield Horizontal(label, input_widget)

            # Prediction section
            with Vertical(classes="section") as pred_section:
                pred_section.border_title = "🎯 Prediction"
                self.prediction_label = Label(
                    "Predicted Performance Index: —",
                    id=f"prediction-{sanitize_id(self.model_name)}",
                )
                yield self.prediction_label

    def on_mount(self) -> None:
        # Set initial values if provided
        for feat, widget in self.inputs.items():
            if feat in self.initial_values:
                val = self.initial_values[feat]
                if "Activities" in feat:
                    widget.value = str(int(val))
                else:
                    widget.value = str(val)
        # Trigger initial prediction
        self.predict()

    def predict(self):
        try:
            values = []
            for feat in self.feature_names:
                val = self.inputs[feat].value.strip()
                if not val:
                    raise ValueError("Missing input")
                values.append(float(val))
            x = np.array(values)
            pred = self.coeffs[0] + np.dot(self.coeffs[1:], x)
            self.prediction_label.update(f"Performance Index: {pred:.2f}")
        except Exception:
            self.prediction_label.update("Performance Index: —")

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed):
        self.predict()
