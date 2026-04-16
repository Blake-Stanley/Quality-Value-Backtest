"""
make_sector_pie.py
Generates a side-by-side sector allocation pie chart for the EW long and short books
from the holdings snapshot. Saves to Output/Charts/sector_pie.png.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

HOLDINGS_PATH = Path("Output/holdings_snapshot.xlsx")
OUTPUT_PATH   = Path("Output/Charts/sector_pie.png")
BG_COLOR      = "#bedeff"


def parse_book(xl: pd.ExcelFile, sheet: str) -> pd.Series:
    df = xl.parse(sheet, header=None)
    df.columns = df.iloc[4]
    df = df.iloc[5:].reset_index(drop=True)
    df = df[df["Rank"].notna() & (df["Rank"] != "Rank")]
    return df["Sector"].value_counts()


def draw_pie(ax, counts, colors, title):
    ax.set_facecolor(BG_COLOR)
    wedges, _, autotexts = ax.pie(
        counts,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
    ax.set_title(title, color="black", fontsize=13, fontweight="bold", pad=14)


def main():
    xl = pd.ExcelFile(HOLDINGS_PATH)
    long_counts  = parse_book(xl, "Long Book")
    short_counts = parse_book(xl, "Short Book")

    all_sectors = sorted(set(long_counts.index) | set(short_counts.index))
    palette = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
        "#E377C2", "#7F7F7F",
    ]
    color_map   = {s: palette[i % len(palette)] for i, s in enumerate(all_sectors)}
    long_colors  = [color_map[s] for s in long_counts.index]
    short_colors = [color_map[s] for s in short_counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)

    draw_pie(axes[0], long_counts,  long_colors,  "Long Book  (EW, Dec 2024)")
    draw_pie(axes[1], short_counts, short_colors, "Short Book  (EW, Dec 2024)")

    legend_handles = [mpatches.Patch(color=color_map[s], label=s) for s in all_sectors]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        labelcolor="black",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Sector Allocation — Long/Short Portfolio",
        color="black", fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
