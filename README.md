2026-02-11 작업 v1: 바이낸스 선물 과거데이터 캐시 + 지표(YAML 선택) + 4시간치 리플레이 윈도우 + 에이전트 인터페이스 뼈대.
기본은 fapi를 먼저 시도하고 막히면 Binance Vision 정적 데이터로 자동 폴백됩니다.
Step2 룰에이전트 예시: `python examples/step2_backtest_ruleagent.py`
Step2 더미에이전트(항상 거래) 예시: `python examples/step2_backtest_dummyagent.py`

```bash
pip install -e .[dev]
python examples/step1_smoketest.py
pytest -q
```
