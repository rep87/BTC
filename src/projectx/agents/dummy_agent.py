from __future__ import annotations

from projectx.agents.interface import AgentInput, AgentOutput, BaseAgent


class DummyAgent(BaseAgent):
    def __init__(
        self,
        open_step: int = 1,
        close_step: int = 2,
        side: str = "long",
        size_pct: float = 0.5,
        leverage: int = 2,
    ):
        self.open_step = open_step
        self.close_step = close_step
        self.side = side.lower() if side.lower() in {"long", "short"} else "long"
        self.size_pct = size_pct
        self.leverage = leverage
        self.step_count = 0

    def run(self, input_data: AgentInput) -> AgentOutput:
        self.step_count += 1
        if self.step_count == self.open_step:
            return AgentOutput(
                decision={
                    "action": self.side,
                    "size_pct": self.size_pct,
                    "leverage": self.leverage,
                    "reason": f"open at step {self.step_count}",
                },
                notes=["dummy open"],
            )
        if self.step_count == self.close_step:
            return AgentOutput(
                decision={"action": "close", "reason": f"close at step {self.step_count}"},
                notes=["dummy close"],
            )
        return AgentOutput(decision={"action": "hold", "reason": f"hold at step {self.step_count}"}, notes=["dummy hold"])
