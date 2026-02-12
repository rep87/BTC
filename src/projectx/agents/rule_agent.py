from __future__ import annotations

from projectx.agents.interface import AgentInput, AgentOutput


class RuleAgent:
    """Simple deterministic rule-based agent for Step1/Step2 demos."""

    def run(self, input_data: AgentInput) -> AgentOutput:
        if input_data.new_chunk_df.empty:
            return AgentOutput(
                decision={"action": "hold", "confidence": 0, "size_pct": 0.1, "leverage": 1, "reason": "no new data"},
                notes=["no new data"],
                strategy_patch=None,
            )

        chunk = input_data.new_chunk_df
        latest_close = float(chunk["close"].iloc[-1])
        rsi_cols = sorted([col for col in chunk.columns if col.startswith("rsi_")])

        if rsi_cols:
            rsi_col = rsi_cols[0]
            rsi_value = float(chunk[rsi_col].iloc[-1])
            if rsi_value <= 30:
                action = "long"
                confidence = 75
                reason = f"{rsi_col} oversold"
            elif rsi_value >= 70:
                action = "short"
                confidence = 75
                reason = f"{rsi_col} overbought"
            else:
                action = "hold"
                confidence = 55
                reason = f"{rsi_col} neutral"
            notes = [f"{rsi_col}={rsi_value:.2f}", f"close={latest_close:.2f}"]
        else:
            closes = chunk["close"].tail(6)
            if len(closes) >= 2:
                move = float(closes.iloc[-1] - closes.iloc[0]) / float(closes.iloc[0])
            else:
                move = 0.0

            threshold = 0.002
            if move <= -threshold:
                action = "long"
                confidence = 62
                reason = "down move mean-revert"
            elif move >= threshold:
                action = "short"
                confidence = 62
                reason = "up move mean-revert"
            else:
                action = "hold"
                confidence = 50
                reason = "weak momentum"
            notes = [f"move={move:.4f}", f"close={latest_close:.2f}"]

        return AgentOutput(
            decision={
                "action": action,
                "confidence": int(max(0, min(100, confidence))),
                "size_pct": 0.2,
                "leverage": 1,
                "reason": reason,
            },
            notes=notes[:3],
            strategy_patch=None,
        )
