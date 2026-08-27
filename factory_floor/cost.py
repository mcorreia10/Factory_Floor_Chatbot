"""LLM cost tracking (phase 3).

Every LLM call in the project already funnels through ``rag.get_llm()``. This module
adds a callback that meters token usage on each call, a per-session accumulator, and a
pre-flight daily spend cap. The daily ledger itself lives in ``factory_floor.audit``
(``cost_events`` table) since phase 5 — ``DailyLedger`` is re-exported here so existing
imports keep working.

Nothing here is on by default: with no ``FACTORY_FLOOR_DAILY_SPEND_CAP_USD`` set, the
cap check is a no-op and the only effect is that ``DiagnosticResult.cost`` and the
sidebar cost line get populated.
"""

import logging
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler

from factory_floor.audit import DailyLedger  # re-export (phase 5): the ledger is SQLite now

__all__ = [
    "MODEL_PRICING",
    "PENDING_TURN_ESTIMATE_USD",
    "SpendCapExceeded",
    "count_tokens",
    "estimate_cost",
    "UsageAccumulator",
    "CostTrackingCallback",
    "DailyLedger",
    "check_spend_cap",
]

logger = logging.getLogger("factory_floor.cost")

# USD per 1,000,000 tokens, (input, output). Prices as of 2026-08 from
# https://openai.com/api/pricing/ — this table is the single source of cost math;
# update the values and this date if pricing changes.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}
# Unknown model -> assume gpt-4.1-mini-class rather than zero, so cost is never
# silently under-reported.
_DEFAULT_PRICING = (0.40, 1.60)

# Conservative guess at one diagnostic turn's cost, used only for the pre-flight cap
# check so the cap trips *before* going over rather than one turn after.
PENDING_TURN_ESTIMATE_USD = 0.05


class SpendCapExceeded(RuntimeError):
    """Raised by check_spend_cap when the daily cap would be exceeded."""


# --- token counting --------------------------------------------------------------

_ENCODINGS: dict[str, object] = {}


def _encoding_for(model: str):
    if model not in _ENCODINGS:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        _ENCODINGS[model] = enc
    return _ENCODINGS[model]


def count_tokens(text: str, model: str = "gpt-4.1-mini") -> int:
    if not text:
        return 0
    try:
        return len(_encoding_for(model).encode(text))
    except Exception:  # tiktoken unavailable / registry miss — rough fallback
        return max(1, len(text) // 4)


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    in_price, out_price = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


# --- accumulation --------------------------------------------------------------

@dataclass
class UsageAccumulator:
    """Running token/cost totals. Lives per session in ``st.session_state`` (via
    as_dict/from_dict) and per request inside ``services.run_diagnostic``."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_usd: float = 0.0
    n_calls: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, input_tokens: int, output_tokens: int, model: str) -> None:
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        usd = estimate_cost(input_tokens, output_tokens, model)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_usd += usd
        self.n_calls += 1
        slot = self.by_model.setdefault(
            model, {"input_tokens": 0, "output_tokens": 0, "usd": 0.0, "n_calls": 0}
        )
        slot["input_tokens"] += input_tokens
        slot["output_tokens"] += output_tokens
        slot["usd"] += usd
        slot["n_calls"] += 1

    def as_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_usd": self.total_usd,
            "n_calls": self.n_calls,
            "by_model": self.by_model,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "UsageAccumulator":
        data = data or {}
        return cls(
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_usd=data.get("total_usd", 0.0),
            n_calls=data.get("n_calls", 0),
            by_model=dict(data.get("by_model", {})),
        )

    def merge(self, other: dict | None) -> None:
        """Fold another accumulator's dict into this one — used by app.py to keep a
        session running total across turns."""
        other = other or {}
        self.total_input_tokens += other.get("total_input_tokens", 0)
        self.total_output_tokens += other.get("total_output_tokens", 0)
        self.total_usd += other.get("total_usd", 0.0)
        self.n_calls += other.get("n_calls", 0)
        for model, slot in other.get("by_model", {}).items():
            mine = self.by_model.setdefault(
                model, {"input_tokens": 0, "output_tokens": 0, "usd": 0.0, "n_calls": 0}
            )
            for key in ("input_tokens", "output_tokens", "usd", "n_calls"):
                mine[key] += slot.get(key, 0)


class CostTrackingCallback(BaseCallbackHandler):
    """Reads token usage off each LLM response into an injected UsageAccumulator.

    Prefers ``AIMessage.usage_metadata`` (present on non-streamed calls, and on
    streamed calls when ``stream_usage=True``). Falls back to counting the completion
    text with tiktoken when no usage metadata is available, so every call still
    registers (n_calls increments) even if the token counts are approximate.
    """

    def __init__(self, accumulator: UsageAccumulator, default_model: str = "gpt-4.1-mini"):
        self.accumulator = accumulator
        self.default_model = default_model

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: D401 - callback signature
        input_tokens = 0
        output_tokens = 0
        model = self.default_model
        fallback_text_parts: list[str] = []

        for gen_list in getattr(response, "generations", None) or []:
            for gen in gen_list:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None) if message is not None else None
                if usage:
                    input_tokens += usage.get("input_tokens", 0) or 0
                    output_tokens += usage.get("output_tokens", 0) or 0
                    meta = getattr(message, "response_metadata", None) or {}
                    model = meta.get("model_name") or meta.get("model") or model
                else:
                    fallback_text_parts.append(getattr(gen, "text", "") or "")

        if input_tokens == 0 and output_tokens == 0:
            token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            input_tokens = token_usage.get("prompt_tokens", 0) or 0
            output_tokens = token_usage.get("completion_tokens", 0) or 0
            model = (getattr(response, "llm_output", None) or {}).get("model_name", model)

        if input_tokens == 0 and output_tokens == 0:
            # Last resort: estimate the completion from its text. Input is unknown here
            # (the prompt is not on the response), so it is left at 0 and the turn is
            # under-counted — acceptable, and only hit when usage metadata is absent.
            output_tokens = sum(count_tokens(part, model) for part in fallback_text_parts)

        self.accumulator.add(input_tokens, output_tokens, model)


# --- spend cap ---------------------------------------------------------------------

def _default_alert(daily_total_usd: float, cap: float) -> None:
    logger.warning(
        "LLM spend at $%.2f of the $%.2f daily cap (>= %d%%).",
        daily_total_usd,
        cap,
        int(100 * daily_total_usd / cap) if cap else 0,
    )


def check_spend_cap(
    daily_total_usd: float,
    *,
    cap: float | None,
    pending_estimate: float = 0.0,
    alert_threshold: float = 0.8,
    on_alert=_default_alert,
) -> None:
    """Raise SpendCapExceeded if today's spend plus a conservative estimate for this
    turn would meet or exceed ``cap``. No-op when ``cap`` is None. Fires ``on_alert``
    once the projected total crosses ``alert_threshold * cap``."""
    if cap is None:
        return
    projected = daily_total_usd + pending_estimate
    if projected >= cap:
        raise SpendCapExceeded(
            f"Daily LLM spend cap reached: about ${daily_total_usd:.2f} spent today, "
            f"and this turn is estimated at ~${pending_estimate:.2f}, against a "
            f"${cap:.2f} cap. Raise FACTORY_FLOOR_DAILY_SPEND_CAP_USD or wait until 00:00 UTC."
        )
    if on_alert is not None and projected >= alert_threshold * cap:
        on_alert(daily_total_usd, cap)
