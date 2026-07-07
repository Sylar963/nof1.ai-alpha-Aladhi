"""Performance metrics and plain-text report for simulator results."""

import statistics
from typing import Dict, List


def _max_drawdown_pct(starting_equity: float, equity_curve: List[Dict]) -> float:
    peak = starting_equity
    max_dd = 0.0
    for point in equity_curve:
        eq = point["equity"]
        if eq > peak:
            peak = eq
        elif peak > 0:
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _trade_pnl(trade) -> float:
    return trade["pnl"] if isinstance(trade, dict) else trade.pnl


def _trade_attr(trade, name):
    return trade[name] if isinstance(trade, dict) else getattr(trade, name)


def compute_metrics(result: Dict) -> Dict:
    """Build a metrics report dict from ``Simulator.result()`` output."""
    starting = result["starting_equity"]
    final = result["final_equity"]
    trades = result.get("closed_trades", [])
    pnls = [_trade_pnl(t) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float('inf')
    else:
        profit_factor = 0.0

    holds = [_trade_attr(t, "hold_seconds") for t in trades]

    per_asset: Dict[str, Dict] = {}
    for t in trades:
        asset = _trade_attr(t, "asset")
        bucket = per_asset.setdefault(
            asset, {"trades": 0, "wins": 0, "net_pnl": 0.0, "fees": 0.0}
        )
        pnl = _trade_pnl(t)
        bucket["trades"] += 1
        bucket["wins"] += 1 if pnl > 0 else 0
        bucket["net_pnl"] += pnl
        bucket["fees"] += _trade_attr(t, "fees")
    for bucket in per_asset.values():
        bucket["win_rate"] = bucket["wins"] / bucket["trades"] if bucket["trades"] else 0.0

    return {
        "starting_equity": starting,
        "final_equity": final,
        "total_return_pct": (final - starting) / starting * 100.0 if starting else 0.0,
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "expectancy": sum(pnls) / len(pnls) if pnls else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_pct": _max_drawdown_pct(starting, result.get("equity_curve", [])),
        "total_fees": result.get("fees_paid", 0.0),
        "avg_hold_seconds": sum(holds) / len(holds) if holds else 0.0,
        "median_hold_seconds": statistics.median(holds) if holds else 0.0,
        "per_asset": per_asset,
        "skipped_count": len(result.get("skipped", [])),
        "skipped": result.get("skipped", []),
        "clamp_notes": result.get("clamp_notes", []),
        "open_position_count": len(result.get("open_positions", {})),
    }


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_report(report: Dict) -> str:
    """Render a metrics report dict as a plain-text summary."""
    pf = report["profit_factor"]
    pf_text = "inf" if pf == float('inf') else f"{pf:.2f}"
    lines = [
        "=== Backtest Report ===",
        f"Equity:        ${report['starting_equity']:,.2f} -> ${report['final_equity']:,.2f}",
        f"Total return:  {report['total_return_pct']:+.2f}%",
        f"Trades closed: {report['trade_count']}",
        f"Win rate:      {report['win_rate']*100:.1f}%",
        f"Expectancy:    ${report['expectancy']:+,.2f}/trade",
        f"Profit factor: {pf_text}",
        f"Max drawdown:  {report['max_drawdown_pct']:.2f}%",
        f"Total fees:    ${report['total_fees']:,.2f}",
        f"Avg hold:      {_fmt_duration(report['avg_hold_seconds'])}",
        f"Median hold:   {_fmt_duration(report['median_hold_seconds'])}",
        f"Skipped:       {report['skipped_count']} decision(s)",
        f"Open at end:   {report['open_position_count']} position(s)",
    ]
    if report["per_asset"]:
        lines.append("--- Per asset ---")
        for asset in sorted(report["per_asset"]):
            b = report["per_asset"][asset]
            lines.append(
                f"{asset:>6}: {b['trades']} trade(s), "
                f"win {b['win_rate']*100:.0f}%, "
                f"pnl ${b['net_pnl']:+,.2f}, fees ${b['fees']:,.2f}"
            )
    if report["skipped"]:
        lines.append("--- Skipped decisions ---")
        for s in report["skipped"][:20]:
            lines.append(f"{s['timestamp']} {s['asset']} {s['action']}: {s['reason']}")
    return "\n".join(lines)
