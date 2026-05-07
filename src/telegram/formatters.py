"""
Telegram Message Formatters.

Formats analysis results for Telegram display.
Mobile-optimized, concise output.
"""

from __future__ import annotations


def format_risk_report(report: dict) -> str:
    """
    Format full risk report for Telegram.

    Designed for mobile readability.
    """
    mint = report["mint"]
    metrics = report["metrics"]
    archetypes = report["archetypes"]

    # Risk level based on stability score
    stability = report["stability_score"]
    if stability >= 70:
        risk_emoji = "🟢"
        risk_level = "Low"
    elif stability >= 40:
        risk_emoji = "🟡"
        risk_level = "Medium"
    else:
        risk_emoji = "🔴"
        risk_level = "High"

    # Format archetype distribution
    top_archetypes = sorted(archetypes.items(), key=lambda x: x[1], reverse=True)[:3]
    archetype_str = " | ".join(
        f"{name}: {pct:.0%}" for name, pct in top_archetypes
    )

    msg = f"""
📊 *Token Analysis*
`{mint[:8]}...{mint[-4:]}`

*Risk Level:* {risk_emoji} {risk_level}
*Stability Score:* {stability:.0f}/100

*Distribution Metrics*
├ HHI: {metrics['hhi']:.4f}
├ Gini: {metrics['gini']:.2f}
├ Whale Dominance: {metrics['wdr']:.1%}
└ Churn: {metrics['churn']:.1%}

*Sell Pressure Index:* {report['sell_pressure']:.2f}
*Sybil Probability:* {report['sybil_prob']:.0%}

*Holder Archetypes*
{archetype_str}

*Holders:* {report['holder_count']:,}

---
_All outputs are probabilistic._
_Not trading advice._
"""
    return msg.strip()


def format_holder_summary(data: dict) -> str:
    """Format top holder breakdown."""
    mint = data["mint"]
    holders = data["top_holders"]

    lines = [f"*Top Holders* - `{mint[:8]}...`\n"]

    for h in holders:
        archetype_emoji = {
            "sniper": "⚡",
            "accumulator": "📈",
            "dormant_whale": "🐋",
            "coordinated": "🔗",
            "liquidity_actor": "💧",
            "exchange_linked": "🏦",
            "unknown": "❓",
        }.get(h["archetype"], "❓")

        lines.append(
            f"#{h['rank']} | {h['share']:.1%} | {archetype_emoji} {h['archetype']}"
        )

    lines.append("\n_Archetypes are behavioral, not identity._")

    return "\n".join(lines)


def format_quick_summary(data: dict) -> str:
    """Format 3-line quick summary."""
    mint = data["mint"]
    holders = data["holder_count"]
    stability = data["stability_score"]
    risk = data["risk_level"]

    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")

    return f"""
`{mint[:8]}...` | {holders:,} holders
Stability: {stability}/100 | Risk: {risk_emoji} {risk}
_Use /analyze for full report_
""".strip()


def format_error(error: str) -> str:
    """Format error message."""
    # Sanitize error message
    safe_error = error[:200] if len(error) > 200 else error
    safe_error = safe_error.replace("`", "'")

    return f"""
❌ *Analysis Failed*

Error: `{safe_error}`

Please try again or use /help for assistance.
""".strip()


def format_coordination_alert(
    mint: str,
    cluster_count: int,
    coordination_score: float,
) -> str:
    """Format coordination alert message."""
    return f"""
⚠️ *Coordination Alert*
Token: `{mint[:8]}...`

Detected {cluster_count} potential coordinated wallet clusters.
Coordination Score: {coordination_score:.0%}

This may indicate:
- Sybil attacks
- Coordinated trading
- Wash trading

_This is not a definitive fraud label._
""".strip()
