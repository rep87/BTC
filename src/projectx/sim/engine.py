from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from projectx.agents.interface import AgentInput, BaseAgent
from projectx.replay.window import ReplayWindow
from projectx.sim.metrics import compute_metrics
from projectx.sim.types import Fill, Position, SimConfig, StepDecision, clamp_leverage, clamp_size_pct


def _map_action(raw_action: str) -> str:
    action = str(raw_action).lower()
    mapping = {
        "risk_on": "long",
        "risk_off": "short",
        "observe_up": "long",
        "observe_down": "short",
    }
    return mapping.get(action, action)


def _copy_position(position: Position) -> Position:
    return Position(
        side=position.side,
        qty=position.qty,
        entry_price=position.entry_price,
        leverage=position.leverage,
        margin=position.margin,
        entry_time=position.entry_time,
    )


class BacktestEngine:
    def __init__(self, df_eval: pd.DataFrame, config: SimConfig):
        self.df_eval = df_eval.copy().sort_index()
        self.df_eval.index = pd.to_datetime(self.df_eval.index, utc=True)
        self.config = config
        self.position = Position()
        self.cash = float(config.start_equity)
        self.status = "ok"
        self.fills: list[Fill] = []
        self._state_events: list[tuple[pd.Timestamp, float, Position]] = []

    def run(self, window: ReplayWindow, agent: BaseAgent) -> dict[str, Any]:
        window.reset()
        eval_df = self.df_eval[(self.df_eval.index >= window.eval_start) & (self.df_eval.index < window.eval_end)].copy()
        if eval_df.empty:
            empty_curve = pd.DataFrame(columns=["equity", "drawdown", "position_side", "price"])
            return {"fills": [], "equity_curve": empty_curve, "metrics": compute_metrics(empty_curve, [], "ok")}

        indicator_cols = [
            c
            for c in eval_df.columns
            if c.startswith("ema_") or c.startswith("rsi_") or c.startswith("atr_") or c in {"macd", "macd_signal", "macd_hist"}
        ]

        while not window.is_done() and self.status == "ok":
            chunk = window.step()
            if chunk.empty:
                continue

            ai_input = AgentInput(
                context_df=window.get_context(),
                revealed_df=window.get_revealed(),
                new_chunk_df=chunk,
                metadata={"step_hours": self.config.step_hours},
                indicators_enabled=indicator_cols,
            )
            output = agent.run(ai_input)
            decision = self._parse_decision(output.decision)

            decision_time = chunk.index.max()
            decision_price = float(chunk["close"].iloc[-1])
            equity_at_decision = self._equity_at_price(decision_price)

            next_rows = eval_df[eval_df.index > decision_time]
            if next_rows.empty:
                continue

            fill_time = next_rows.index[0]
            fill_price = float(next_rows.iloc[0]["open"])
            self._execute(decision, fill_time, fill_price, equity_at_decision)

            if not self.config.allow_negative_equity and self._equity_at_price(fill_price) <= 0:
                self.status = "bankrupt"
                break

        equity_curve = self._build_equity_curve(eval_df)
        metrics = compute_metrics(equity_curve, [asdict(f) for f in self.fills], self.status)
        return {"fills": self.fills, "equity_curve": equity_curve, "metrics": metrics}

    def _parse_decision(self, decision: dict[str, Any]) -> StepDecision:
        action = _map_action(str(decision.get("action", "hold")))
        if action not in {"hold", "long", "short", "close"}:
            action = "hold"
        size_pct = clamp_size_pct(decision.get("size_pct"), self.config.default_size_pct)
        leverage = clamp_leverage(
            decision.get("leverage"),
            self.config.min_leverage,
            self.config.max_leverage,
            self.config.default_leverage,
        )
        return StepDecision(action=action, size_pct=size_pct, leverage=leverage)

    def _equity_at_price(self, price: float) -> float:
        if self.position.side == "flat" or self.position.qty <= 0:
            return self.cash
        if self.position.side == "long":
            upnl = self.position.qty * (price - self.position.entry_price)
        else:
            upnl = self.position.qty * (self.position.entry_price - price)
        return self.cash + upnl

    def _target_qty(self, equity: float, size_pct: float, leverage: int, price: float) -> tuple[float, float]:
        margin = max(0.0, equity * size_pct)
        notional = margin * leverage
        qty = 0.0 if price <= 0 else notional / price
        return qty, margin

    def _record_fill(self, fill: Fill, position_after: Position) -> None:
        self.fills.append(fill)
        self._state_events.append((pd.Timestamp(fill.time), self.cash, _copy_position(position_after)))

    def _close_position(self, fill_time: pd.Timestamp, fill_price: float) -> None:
        if self.position.side == "flat" or self.position.qty <= 0:
            return

        if self.position.side == "long":
            pnl = self.position.qty * (fill_price - self.position.entry_price)
            side = "long"
        else:
            pnl = self.position.qty * (self.position.entry_price - fill_price)
            side = "short"

        notional = self.position.qty * fill_price
        fee = self.config.fee_rate * notional
        self.cash += pnl - fee
        closed_qty = self.position.qty
        self.position = Position()
        self._record_fill(
            Fill(
                time=fill_time.isoformat(),
                side=side,
                action="close",
                price=fill_price,
                qty=closed_qty,
                notional=notional,
                fee=fee,
                equity_after=self.cash,
            ),
            self.position,
        )

    def _open_position(self, side: str, qty: float, leverage: int, margin: float, fill_time: pd.Timestamp, fill_price: float, action: str) -> None:
        if qty <= 0:
            return
        notional = qty * fill_price
        fee = self.config.fee_rate * notional
        self.cash -= fee
        self.position = Position(
            side=side,
            qty=qty,
            entry_price=fill_price,
            leverage=leverage,
            margin=margin,
            entry_time=fill_time.isoformat(),
        )
        self._record_fill(
            Fill(
                time=fill_time.isoformat(),
                side=side,
                action=action,
                price=fill_price,
                qty=qty,
                notional=notional,
                fee=fee,
                equity_after=self.cash,
            ),
            self.position,
        )

    def _resize_position(self, target_qty: float, target_leverage: int, target_margin: float, fill_time: pd.Timestamp, fill_price: float) -> None:
        current_qty = self.position.qty
        delta_qty = target_qty - current_qty
        if abs(delta_qty) < 1e-12:
            return

        fee = self.config.fee_rate * abs(delta_qty) * fill_price
        self.cash -= fee

        if delta_qty > 0:
            new_qty = current_qty + delta_qty
            weighted_entry = ((current_qty * self.position.entry_price) + (delta_qty * fill_price)) / new_qty
            self.position.qty = new_qty
            self.position.entry_price = weighted_entry
        else:
            reduce_qty = abs(delta_qty)
            if self.position.side == "long":
                realized = reduce_qty * (fill_price - self.position.entry_price)
            else:
                realized = reduce_qty * (self.position.entry_price - fill_price)
            self.cash += realized
            self.position.qty = max(0.0, current_qty - reduce_qty)
            if self.position.qty <= 0:
                self.position = Position()
                return

        self.position.leverage = target_leverage
        self.position.margin = target_margin
        self._record_fill(
            Fill(
                time=fill_time.isoformat(),
                side=self.position.side,
                action="resize",
                price=fill_price,
                qty=self.position.qty,
                notional=self.position.qty * fill_price,
                fee=fee,
                equity_after=self.cash,
            ),
            self.position,
        )

    def _execute(self, decision: StepDecision, fill_time: pd.Timestamp, fill_price: float, equity_at_decision: float) -> None:
        if decision.action == "hold":
            return
        if decision.action == "close":
            self._close_position(fill_time, fill_price)
            return

        assert decision.size_pct is not None
        assert decision.leverage is not None
        target_side = decision.action
        target_qty, target_margin = self._target_qty(
            equity=equity_at_decision,
            size_pct=decision.size_pct,
            leverage=decision.leverage,
            price=fill_price,
        )
        if target_qty <= 0:
            return

        if self.position.side == "flat":
            self._open_position(target_side, target_qty, decision.leverage, target_margin, fill_time, fill_price, "open")
            return

        if self.position.side == target_side:
            self._resize_position(target_qty, decision.leverage, target_margin, fill_time, fill_price)
            return

        self._close_position(fill_time, fill_price)
        self._open_position(target_side, target_qty, decision.leverage, target_margin, fill_time, fill_price, "flip")

    def _build_equity_curve(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        events: dict[pd.Timestamp, list[tuple[float, Position]]] = {}
        for ts, cash_after, pos_after in self._state_events:
            events.setdefault(ts, []).append((cash_after, pos_after))

        cash = float(self.config.start_equity)
        position = Position()
        rows: list[dict[str, Any]] = []

        for ts, row in eval_df.iterrows():
            for cash_after, pos_after in events.get(ts, []):
                cash = cash_after
                position = _copy_position(pos_after)

            close_price = float(row["close"])
            if position.side == "flat" or position.qty <= 0:
                equity = cash
            elif position.side == "long":
                equity = cash + position.qty * (close_price - position.entry_price)
            else:
                equity = cash + position.qty * (position.entry_price - close_price)

            rows.append({"timestamp": ts, "equity": equity, "position_side": position.side, "price": close_price})

            if not self.config.allow_negative_equity and equity <= 0:
                self.status = "bankrupt"
                break

        curve = pd.DataFrame(rows).set_index("timestamp")
        if curve.empty:
            curve["drawdown"] = []
            return curve

        running_max = curve["equity"].cummax()
        curve["drawdown"] = (curve["equity"] / running_max) - 1.0
        return curve
