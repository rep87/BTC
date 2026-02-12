from __future__ import annotations

from projectx.sim.types import SimConfig, StepDecision, clamp_leverage, clamp_size_pct

ACTION_ALIASES = {
    "buy": "long",
    "sell": "short",
    "exit": "close",
}
VALID_ACTIONS = {"hold", "long", "short", "close"}


def normalize_decision(decision_dict: dict, config: SimConfig) -> tuple[StepDecision, str | None]:
    raw_action = str(decision_dict.get("action", "hold")).strip().lower()
    mapped = ACTION_ALIASES.get(raw_action, raw_action)
    warning: str | None = None
    if mapped not in VALID_ACTIONS:
        warning = f"Unknown action '{raw_action}' mapped to hold"
        mapped = "hold"

    size_pct = clamp_size_pct(decision_dict.get("size_pct"), config.default_size_pct)
    leverage = clamp_leverage(
        decision_dict.get("leverage"),
        config.min_leverage,
        config.max_leverage,
        config.default_leverage,
    )
    return StepDecision(action=mapped, size_pct=size_pct, leverage=leverage), warning
