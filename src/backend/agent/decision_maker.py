"""Decision-making agent that orchestrates LLM prompts and indicator lookups."""

import requests
from src.backend.config_loader import CONFIG
from src.backend.indicators.taapi_client import TAAPIClient
import json
import logging
from datetime import datetime

class TradingAgent:
    """High-level trading agent that delegates reasoning to an LLM service."""

    def __init__(self):
        """Initialize LLM configuration, metadata headers, and indicator helper."""
        self.model = CONFIG["llm_model"]
        self.api_key = CONFIG["openrouter_api_key"]
        base = CONFIG["openrouter_base_url"]
        self.base_url = f"{base}/chat/completions"
        self.referer = CONFIG.get("openrouter_referer")
        self.app_title = CONFIG.get("openrouter_app_title")
        # Fast/cheap sanitizer model to normalize outputs on parse failures
        self.sanitize_model = CONFIG.get("sanitize_model") or "openai/gpt-5"

    def decide_trade(self, assets, context):
        """Decide for multiple assets in one call.

        Args:
            assets: Iterable of asset tickers to score.
            context: Structured market/account state forwarded to the LLM.

        Returns:
            List of trade decision payloads, one per asset.
        """
        return self._decide(context, assets=assets)

    def _decide(self, context, assets):
        """Dispatch decision request to the LLM and enforce output contract.

        This agent is PERPS-ONLY. Options reasoning runs through a separate
        ``OptionsAgent`` on its own cadence with its own prompt + schema —
        mixing the two in a single prompt consistently caused cross-venue
        contamination (the perps LLM would emit options strategies without
        setting ``venue='thalex'``, so they'd be routed through the perps
        path and never become actionable options proposals).
        """
        system_prompt = (
            "You are a rigorous QUANTITATIVE TRADER and interdisciplinary MATHEMATICIAN-ENGINEER optimizing risk-adjusted returns on PERPETUAL FUTURES (Hyperliquid) under real execution, margin, and funding constraints.\n"
            "SCOPE: This agent ONLY trades perps. Do NOT emit options strategies, option legs, or Thalex fields — options decisions are produced by a separate specialist agent and merged downstream. If options are the right tool for a situation, HOLD on perps and let the options agent act.\n"
            "POSITION-AWARE EXECUTION (critical — read carefully):\n"
            "- The bot auto-cancels ALL open orders for an asset before placing a new order.\n"
            "- allocation_usd is the DESIRED NEW POSITION size, NOT the total order size.\n"
            "- If you have an existing short and you choose action='buy', the bot will FIRST close the short at its full size, THEN open a long for allocation_usd. You do NOT need to add the short size into allocation_usd yourself.\n"
            "- If you have an existing long and you choose action='sell', same logic: bot closes the long first, then opens a short for allocation_usd.\n"
            "- If you ONLY want to close a position (flatten to zero, no new exposure), set allocation_usd=0 with the opposite action (e.g., action='buy' with allocation_usd=0 to close a short).\n"
            "- The context includes positions with signed 'quantity' (negative = short, positive = long). Use this to decide your action.\n"
        )
        system_prompt += (
            "You will receive market + account context for SEVERAL assets, including:\n"
            f"- assets = {json.dumps(assets)}\n"
            "- per-asset intraday (5m) metrics centered on SMA99, Keltner(130,4), AVWAP anchored at 2026-01-01 00:00 UTC, and opening range\n"
            "- per-asset higher-timeframe metrics centered on SMA99, Keltner(130,4), and the same anchored AVWAP\n"
            "- Active Trades with Exit Plans\n"
            "- Recent Trading History\n\n"
            "Always use the 'current time' provided in the user message to evaluate any time-based conditions, such as cooldown expirations or timed exit plans.\n\n"
            "Your goal: make decisive, first-principles decisions per asset that minimize churn while capturing edge.\n\n"
            "Aggressively pursue setups where calculated risk is outweighed by expected edge; size positions so downside is controlled while upside remains meaningful.\n\n"
            "Core policy (low-churn, position-aware)\n"
            "1) Respect prior plans: If an active trade has an exit_plan with explicit invalidation (e.g., “close if price loses 4h SMA99” or “close if 5m re-enters the opening range”), DO NOT close or flip early unless that invalidation (or a stronger one) has occurred.\n"
            "2) Hysteresis: Require stronger evidence to CHANGE a decision than to keep it. Only flip direction if BOTH:\n"
            "   a) Higher-timeframe structure supports the new direction (e.g., price relative to SMA99, anchored AVWAP, and Keltner position), AND\n"
            "   b) Intraday structure confirms with opening-range behavior plus anchored AVWAP / Keltner alignment.\n"
            "   Otherwise, prefer HOLD or adjust TP/SL.\n"
            "3) Cooldown: After opening, adding, reducing, or flipping, impose a self-cooldown of at least 3 bars of the decision timeframe (e.g., 3×5m = 15m) before another direction change, unless a hard invalidation occurs. Encode this in exit_plan (e.g., “cooldown_bars:3 until 2025-10-19T15:55Z”). You must honor your own cooldowns on future cycles.\n"
            "4) Funding is a tilt, not a trigger: Do NOT open/close/flip solely due to funding unless expected funding over your intended holding horizon meaningfully exceeds the technical edge. Consider that funding accrues discretely and slowly relative to 5m bars.\n"
            "5) Extension alone is not reversal: price pressing a Keltner extreme is not enough by itself. You still need opening-range and VWAP confirmation to fade or flip.\n"
            "6) Prefer adjustments over exits: If the thesis weakens but is not invalidated, first consider: tighten stop to a recent structure level, trail TP, or reduce size. Flip only on hard invalidation + fresh confluence.\n\n"
            "Decision discipline (per asset)\n"
            "- Choose one: buy / sell / hold.\n"
            "- Proactively harvest profits when price action presents a clear, high-quality opportunity that aligns with your thesis.\n"
            "- You control allocation_usd.\n"
            "- TP/SL sanity:\n"
            "  • BUY: tp_price > current_price, sl_price < current_price\n"
            "  • SELL: tp_price < current_price, sl_price > current_price\n"
            "  If sensible TP/SL cannot be set, use null and explain the logic.\n"
            "- exit_plan must include at least ONE explicit invalidation trigger and may include cooldown guidance you will follow later.\n\n"
            "Leverage policy (perpetual futures)\n"
            "- YOU CAN USE LEVERAGE, ATLEAST 3X LEVERAGE TO GET BETTER RETURN, KEEP IT WITHIN 10X IN TOTAL\n"
            "- In high volatility (very wide Keltner envelope) or during funding spikes, reduce or avoid leverage.\n"
            "- Treat allocation_usd as notional exposure; keep it consistent with safe leverage and available margin.\n"
            "\n"
            "Hard margin constraint (DO NOT VIOLATE)\n"
            "- The account context includes `account.buying_power` with `free_margin`, `withdrawable`, and `max_new_notional_by_asset` (a per-asset cap = conservative available × venue max leverage, already including a 5% fee/slippage buffer).\n"
            "- `allocation_usd` for any action other than `hold` MUST be ≤ `account.buying_power.max_new_notional_by_asset[asset]`. Proposals above this cap will be rejected by the pre-trade guard and logged as `proposal_skipped_insufficient_margin` in the next cycle's `recent_diary` — that is wasted edge and should not happen.\n"
            "- If `account.buying_power.free_margin` and `account.buying_power.withdrawable` are both ≤ 0 for every asset, output `action='hold'` across the board and say so in the rationale. Do not propose a trade the account cannot cover.\n"
            "- The only exception is pure closing (flatten an existing position with no new exposure): set `allocation_usd=0` with the opposite action. Closing consumes margin used, not free margin.\n\n"
            "Reasoning recipe (first principles)\n"
            "- Structure (price vs SMA99, anchored AVWAP, Keltner location, opening-range acceptance/rejection), Liquidity/volatility (Keltner width), Positioning tilt (funding, OI).\n"
            "- Favor alignment across 4h and 5m. Counter-trend scalps require stronger intraday confirmation and tighter risk.\n\n"
            "Output contract\n"
            "- Output a STRICT JSON object with exactly two properties in this order:\n"
            "  • reasoning: long-form string capturing detailed, step-by-step analysis (be verbose).\n"
            "  • trade_decisions: array ordered to match the provided assets list.\n"
            "- Each item REQUIRES: asset, action, allocation_usd, tp_price, sl_price, exit_plan, rationale.\n"
            "- Do NOT emit a 'venue' field, 'strategy', 'legs', 'contracts', or any other options-related field. This agent only produces Hyperliquid perp decisions; downstream routing assumes that.\n"
            "- Do not emit Markdown or any extra properties.\n"
        )
        user_prompt = context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.app_title:
            headers["X-Title"] = self.app_title

        def _post(payload):
            """Send a POST request to OpenRouter, logging request and response metadata."""
            # Log the full request payload for debugging
            logging.info("Sending request to OpenRouter (model: %s)", payload.get('model'))
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {payload.get('model')}\n")
                f.write(f"Headers: {json.dumps({k: v for k, v in headers.items() if k != 'Authorization'})}\n")
                f.write(f"Payload:\n{json.dumps(payload, indent=2)}\n")
            # Tightened from 60s → 45s. The call runs inside asyncio.to_thread
            # at the bot_engine call sites, so a hang would tie up a worker
            # thread (uncancellable from the outer loop) until this fires.
            # 45s is still well above OpenRouter p99 latency.
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=45)
            logging.info("Received response from OpenRouter (status: %s)", resp.status_code)
            if resp.status_code != 200:
                logging.error("OpenRouter error: %s - %s", resp.status_code, resp.text)
                with open("llm_requests.log", "a", encoding="utf-8") as f:
                    f.write(f"ERROR Response: {resp.status_code} - {resp.text}\n")
            resp.raise_for_status()
            return resp.json()

        _SANITIZE_MAX_CHARS = 100_000  # ~25K tokens — plenty for a valid decision payload

        def _sanitize_output(raw_content: str, assets_list):
            """Coerce arbitrary LLM output into the required reasoning + decisions schema."""
            if len(raw_content) > _SANITIZE_MAX_CHARS:
                logging.warning(
                    "Sanitizer input too large (%d chars), truncating to %d",
                    len(raw_content), _SANITIZE_MAX_CHARS,
                )
                raw_content = raw_content[:_SANITIZE_MAX_CHARS]
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "trade_decisions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "asset": {"type": "string", "enum": assets_list},
                                    "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                                    "allocation_usd": {"type": "number"},
                                    "tp_price": {"type": ["number", "null"]},
                                    "sl_price": {"type": ["number", "null"]},
                                    "exit_plan": {"type": "string"},
                                    "rationale": {"type": "string"},
                                },
                                "required": [
                                    "asset", "action", "allocation_usd", "tp_price", "sl_price",
                                    "exit_plan", "rationale",
                                ],
                                "additionalProperties": False,
                            },
                            "minItems": 1,
                        }
                    },
                    "required": ["reasoning", "trade_decisions"],
                    "additionalProperties": False,
                }
                payload = {
                    "model": self.sanitize_model,
                    "messages": [
                        {"role": "system", "content": (
                            "You are a strict JSON normalizer. Return ONLY a JSON array matching the provided JSON Schema. "
                            "If input is wrapped or has prose/markdown, fix it. Do not add fields."
                        )},
                        {"role": "user", "content": raw_content},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "trade_decisions",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                    "temperature": 0,
                }
                resp = _post(payload)
                msg = resp.get("choices", [{}])[0].get("message", {})
                parsed = msg.get("parsed")
                if isinstance(parsed, dict):
                    if "trade_decisions" in parsed:
                        return parsed
                # fallback: try content
                content = msg.get("content") or "[]"
                try:
                    loaded = json.loads(content)
                    if isinstance(loaded, dict) and "trade_decisions" in loaded:
                        return loaded
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    pass
                return {"reasoning": "", "trade_decisions": []}
            except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as se:
                logging.error("Sanitize failed: %s", se)
                return {"reasoning": "", "trade_decisions": []}

        allow_structured = True

        def _build_schema():
            """Assemble the JSON schema used for structured LLM responses.

            Perps-only: no options strategies, no legs, no Thalex fields.
            The schema is ``additionalProperties: false`` so the LLM cannot
            sneak options fields through structured outputs — if the prompt
            ever regresses, the provider rejects the response rather than
            letting a miscategorised options decision reach the bot engine.
            Options decisions flow through ``OptionsAgent`` on its own.
            """
            base_properties = {
                "asset": {"type": "string", "enum": assets},
                "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                "rationale": {"type": "string"},
                "allocation_usd": {"type": ["number", "null"], "minimum": 0},
                "tp_price": {"type": ["number", "null"]},
                "sl_price": {"type": ["number", "null"]},
                "exit_plan": {"type": ["string", "null"]},
            }
            required_keys = [
                "asset", "action", "rationale",
                "allocation_usd", "tp_price", "sl_price", "exit_plan",
            ]
            return {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "trade_decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": base_properties,
                            "required": required_keys,
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    }
                },
                "required": ["reasoning", "trade_decisions"],
                "additionalProperties": False,
            }

        for _ in range(3):
            data = {"model": self.model, "messages": messages}
            _cfg_max = CONFIG.get("llm_max_tokens")
            data["max_tokens"] = int(_cfg_max) if _cfg_max is not None and _cfg_max != "" else 16384
            if allow_structured:
                data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "trade_decisions",
                        "strict": True,
                        "schema": _build_schema(),
                    },
                }
            if CONFIG.get("reasoning_enabled"):
                data["reasoning"] = {
                    "enabled": True,
                    "effort": CONFIG.get("reasoning_effort") or "high",
                    # "max_tokens": CONFIG.get("reasoning_max_tokens") or 100000,
                    "exclude": False,
                }
            if CONFIG.get("provider_config") or CONFIG.get("provider_quantizations"):
                provider_payload = dict(CONFIG.get("provider_config") or {})
                quantizations = CONFIG.get("provider_quantizations")
                if quantizations:
                    provider_payload["quantizations"] = quantizations
                data["provider"] = provider_payload
            try:
                resp_json = _post(data)
            except requests.HTTPError as e:
                try:
                    err = e.response.json()
                except (json.JSONDecodeError, ValueError, AttributeError):
                    err = {}
                # Provider may not support structured outputs / response_format
                err_text = json.dumps(err)
                if allow_structured and ("response_format" in err_text or "structured" in err_text or e.response.status_code in (400, 422)):
                    logging.warning("Provider rejected structured outputs; retrying without response_format.")
                    allow_structured = False
                    continue
                raise

            choices = resp_json.get("choices")
            if not choices:
                error_info = resp_json.get("error", {})
                logging.error("OpenRouter returned no choices: %s", error_info.get("message", resp_json))
                continue
            choice = choices[0]
            message = choice["message"]
            messages.append(message)

            try:
                # Prefer parsed field from structured outputs if present
                if isinstance(message.get("parsed"), dict):
                    parsed = message.get("parsed")
                else:
                    content = message.get("content") or "{}"
                    parsed = json.loads(content)

                if not isinstance(parsed, dict):
                    logging.error("Expected dict payload, got: %s; attempting sanitize", type(parsed))
                    sanitized = _sanitize_output(content if 'content' in locals() else json.dumps(parsed), assets)
                    if sanitized.get("trade_decisions"):
                        return sanitized
                    return {"reasoning": "", "trade_decisions": []}

                reasoning_text = parsed.get("reasoning", "") or ""
                decisions = parsed.get("trade_decisions")

                if isinstance(decisions, list):
                    normalized = []
                    for item in decisions:
                        if isinstance(item, dict):
                            item.setdefault("allocation_usd", 0.0)
                            item.setdefault("tp_price", None)
                            item.setdefault("sl_price", None)
                            item.setdefault("exit_plan", "")
                            item.setdefault("rationale", "")
                            normalized.append(item)
                        elif isinstance(item, list) and len(item) >= 7:
                            normalized.append({
                                "asset": item[0],
                                "action": item[1],
                                "allocation_usd": float(item[2]) if item[2] else 0.0,
                                "tp_price": float(item[3]) if item[3] and item[3] != "null" else None,
                                "sl_price": float(item[4]) if item[4] and item[4] != "null" else None,
                                "exit_plan": item[5] if len(item) > 5 else "",
                                "rationale": item[6] if len(item) > 6 else ""
                            })
                    return {"reasoning": reasoning_text, "trade_decisions": normalized}

                logging.error("trade_decisions missing or invalid; attempting sanitize")
                sanitized = _sanitize_output(content if 'content' in locals() else json.dumps(parsed), assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {"reasoning": reasoning_text, "trade_decisions": []}
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logging.error("JSON parse error: %s, content: %s", e, content[:200])
                # Try sanitizer as last resort
                sanitized = _sanitize_output(content, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {
                    "reasoning": "Parse error",
                    "trade_decisions": [{
                        "asset": a,
                        "action": "hold",
                        "allocation_usd": 0.0,
                        "tp_price": None,
                        "sl_price": None,
                        "exit_plan": "",
                        "rationale": "Parse error"
                    } for a in assets]
                }

        return {
            "reasoning": "retry cap",
            "trade_decisions": [{
                "asset": a,
                "action": "hold",
                "allocation_usd": 0.0,
                "tp_price": None,
                "sl_price": None,
                "exit_plan": "",
                "rationale": "retry cap"
            } for a in assets]
        }
