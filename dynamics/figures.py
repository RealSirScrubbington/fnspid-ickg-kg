"""Thesis figures — every figure regenerates with one command (reproducibility appendix).

Reads the saved outputs of the analysis pipeline (theme lifeline CSVs, formation series) and
writes vector PDFs styled for the dissertation into figures/.

Run: `.venv/Scripts/python -m dynamics.figures`   (KG_CORE_PATH selects the core)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
THEMES = Path("data/dynamics/themes")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "pdf.fonttype": 42,
})

GOLD = [  # (label, event week) — the frozen gold list of themes_eval
    ("COVID crisis", "2020-02-24"), ("Oil crash", "2020-04-20"),
    ("Vaccine race", "2020-05-15"), ("Vaccine efficacy", "2020-11-09"),
    ("GameStop", "2021-01-25"), ("Ukraine", "2022-02-21"),
    ("FTX", "2022-11-07"), ("ChatGPT", "2022-11-28"),
    ("SVB", "2023-03-06"), ("UAW", "2023-09-11"),
]


def lifeline_timeline(mode="lenient", top=28):
    """Horizontal-bar timeline of the top theme lifelines with gold events overlaid."""
    lf = pd.read_csv(THEMES / f"theme_lifelines_{mode}.csv", parse_dates=["birth", "last"])
    lf = lf.sort_values("peak_score", ascending=False).head(top)
    lf = lf.sort_values("birth").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.3, 6.6))
    for y, r in lf.iterrows():
        emerging = r["status"] == "EMERGING"
        col = "#1a5276" if emerging else "#b3b3b3"
        end = max(r["last"], r["birth"] + pd.Timedelta(weeks=1))   # 1-week lifelines stay visible
        ax.plot([r["birth"], end], [y, y], lw=3.2, color=col,
                solid_capstyle="butt", alpha=0.9, zorder=3)
        ax.plot(r["birth"], y, marker="o", ms=3.4, color=col, zorder=4)
        label = " | ".join(str(r["members"]).split(" | ")[:2])
        ax.annotate(label[:34], (end, y), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=6.4, color="#333333")

    for label, wk in GOLD:
        x = pd.Timestamp(wk)
        ax.axvline(x, color="#c0392b", lw=0.7, ls="--", alpha=0.65, zorder=1)
        ax.annotate(label, (x, top - 0.2), rotation=90, va="top", ha="right",
                    fontsize=6.2, color="#c0392b", alpha=0.9)

    ax.set_ylim(-1, top)
    ax.set_yticks([])
    ax.set_xlim(pd.Timestamp("2017-06-01"), pd.Timestamp("2024-06-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines[["left", "top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#1a5276", lw=3, label="EMERGING lifeline (birth $\\bullet$)"),
        Line2D([], [], color="#b3b3b3", lw=3, label="recurring lifeline"),
        Line2D([], [], color="#c0392b", lw=0.8, ls="--", label="gold-list event week"),
    ], loc="upper left", frameon=False)
    ax.set_title(f"Theme lifelines ({mode} configuration): top {top} by peak score")
    fig.tight_layout()
    fig.savefig(OUT / f"timeline_{mode}.pdf")
    fig.savefig(OUT / f"timeline_{mode}.png", dpi=200)
    print(f"wrote figures/timeline_{mode}.pdf (+.png)  [{len(lf)} lifelines]")


DETECTIONS = [  # (gold label, event wk, birth wk or None) — Table 4.2 of the thesis
    ("COVID market crisis", "2020-02-24", "2020-02-03"),
    ("Oil crash (negative WTI)", "2020-04-20", "2020-04-06"),
    ("Vaccine race (Warp Speed)", "2020-05-15", "2020-03-16"),
    ("Vaccine efficacy readout", "2020-11-09", "2020-09-28"),
    ("GameStop squeeze", "2021-01-25", "2021-01-25"),
    ("Russia--Ukraine war", "2022-02-21", "2022-02-28"),
    ("FTX collapse", "2022-11-07", None),
    ("AI / ChatGPT wave", "2022-11-28", None),
    ("SVB banking crisis", "2023-03-06", "2023-03-13"),
    ("UAW strike", "2023-09-11", "2023-09-18"),
]

BONUS = [  # correctly dated discoveries beyond the gold list
    ("Cannabis boom (HEXO/Aurora/Tilray)", "2019-03-18"),
    ("EV/SPAC wave (Nikola/Nio)", "2020-06-22"),
    ("Stay-at-home complex (Zoom/Kroger)", "2020-04-06"),
]


def lead_time_figure():
    """One row per gold event: event week (red), detection birth (blue), lead drawn explicitly."""
    rows = DETECTIONS[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for y, (label, ew, bw) in enumerate(rows):
        e = pd.Timestamp(ew)
        if bw is None:
            ax.plot(e, y, marker="x", ms=7, color="#8c8c8c", mew=1.6, zorder=3)
            ax.annotate("not detected (corpus coverage)", (e, y), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=7, color="#8c8c8c")
        else:
            b = pd.Timestamp(bw)
            lead = int((b - e).days / 7)
            ax.plot([b, e], [y, y], lw=1.4, color="#1a5276", alpha=0.8, zorder=2)
            ax.plot(b, y, marker="o", ms=5.5, color="#1a5276", zorder=4)
            ax.plot(e, y, marker="D", ms=4.5, color="#c0392b", zorder=4)
            side = -1 if lead <= 0 else 1
            ax.annotate(f"{lead:+d} wk", (min(b, e), y), xytext=(-6 if side < 0 else -6, 7),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=7.2, color="#1a5276")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(pd.Timestamp("2019-11-01"), pd.Timestamp("2024-02-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#c0392b", marker="D", ls="", ms=5, label="event week"),
        Line2D([], [], color="#1a5276", marker="o", ls="", ms=5, label="theme birth (live flag)"),
        Line2D([], [], color="#8c8c8c", marker="x", ls="", ms=6, label="miss"),
    ], loc="upper left", frameon=False, fontsize=7.5)
    ax.set_title("Gold-list detection: theme birth vs.\\ event week (median lead $-1$ wk)")
    fig.tight_layout()
    fig.savefig(OUT / "lead_times.pdf")
    fig.savefig(OUT / "lead_times.png", dpi=200)
    print(f"wrote figures/lead_times.pdf (+.png)")


if __name__ == "__main__":
    lifeline_timeline("lenient")
    lead_time_figure()
    from dynamics import figures2
    figures2.main()   # audit-era figures + the rewritten detector overlay (one-command regen)
