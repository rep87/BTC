from __future__ import annotations

from projectx.agents.interface import AgentInput, AgentOutput


class RuleAgent:
    """Very simple deterministic placeholder agent for Step 1."""

    def run(self, input_data: AgentInput) -> AgentOutput:
        if input_data.new_chunk_df.empty:
            return AgentOutput(
                decision={"regime": "unknown", "action": "hold", "confidence": 0},
                notes=["no new data"],
                strategy_patch=None,
            )

        chunk = input_data.new_chunk_df
        latest_close = float(chunk["close"].iloc[-1])

        rsi_cols = [col for col in chunk.columns if col.startswith("rsi_")]
        if rsi_cols:
            rsi_col = sorted(rsi_cols)[0]
            rsi_value = float(chunk[rsi_col].iloc[-1])
            if rsi_value >= 70:
                action = "risk_off"
                regime = "overbought"
                confidence = 75
            elif rsi_value <= 30:
                action = "risk_on"
                regime = "oversold"
                confidence = 75
            else:
                action = "hold"
                regime = "neutral"
                confidence = 55
            notes = [f"{rsi_col}={rsi_value:.2f}", f"close={latest_close:.2f}"]
        else:
            first_close = float(chunk["close"].iloc[0])
            up = latest_close >= first_close
            action = "observe_up" if up else "observe_down"
            regime = "trend_up" if up else "trend_down"
            confidence = 60
            notes = [f"first_close={first_close:.2f}", f"last_close={latest_close:.2f}"]

        return AgentOutput(
            decision={"regime": regime, "action": action, "confidence": confidence},
            notes=notes[:3],
            strategy_patch=None,
        )
