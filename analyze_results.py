import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import contextily as ctx
import numpy as np
import geopandas as gpd
from matplotlib.patches import Polygon as MplPolygon


sns.set_theme(style="darkgrid", context="talk")
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)
IN_DIR = "results"


def day_vs_night():
    file_path = os.path.join(IN_DIR, "results_scenario_2_T17_day_vs_night.csv")
    if not os.path.exists(file_path):
        print("Fișierul nu a fost găsit")
        return

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print("Fișierul este gol")
        return

    df['time_of_day'] = df['time_of_day'].map({'Day': 'Zi', 'Night': 'Noapte'})

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    color_palette = {'Zi': '#f1c40f', 'Noapte': '#2c3e50'}

    sns.barplot(
        data=df,
        x='time_of_day',
        y='evacuation_time',
        hue='time_of_day',
        palette=color_palette,
        errorbar=None,
        edgecolor="black",
        linewidth=2,
        legend=False
    )

    plt.title("Timpul de evacuare pentru căminul T17: zi vs noapte", fontsize=16, fontweight='bold')
    plt.xlabel("Perioada zilei", fontsize=12)
    plt.ylabel("Timp de evacuare (ticks)", fontsize=12)

    plt.xticks(ticks=[0, 1], labels=['Zi (200 de studenți)', 'Noapte (600 de studenți)'])
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Scenario_1_Day_vs_Night.png"), dpi=300)
    plt.close()


def ideal_vs_realistic():
    file_path = os.path.join(IN_DIR, "results_scenario_3_T17_ideal_vs_realistic.csv")
    if not os.path.exists(file_path):
        print("Fișierul nu a fost găsit")
        return

    df = pd.read_csv(file_path)

    max_tick = df[df['people_inside'] > 0]['tick'].max()
    df_plot = df[df['tick'] <= max_tick + 100]

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", context="talk")

    sns.lineplot(
        data=df_plot,
        x='tick',
        y='people_inside',
        hue='mode',
        palette={'ideal': '#2ecc71', 'realistic': '#e74c3c'},
        errorbar=None,
        linewidth=3,
        legend='full'
    )

    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.title("Evacuare ideală vs Evacuare realistă", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Timp (ticks)", fontsize=14)
    plt.ylabel("Studenți în clădire", fontsize=14)

    plt.xlim(0, max_tick + 100)
    plt.ylim(0, df_plot['people_inside'].max() * 1.05)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Ideal (reacție instantanee)', 'Realist (reacție întârziată)'], title="Modul de reacție", loc='upper right', framealpha=1.0)

    plt.ylim(bottom=0)
    plt.xlim(left=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Scenario_3_Ideal_vs_Realistic.png"), dpi=300, bbox_inches='tight')
    plt.close()


def bottleneck():
    file_path = os.path.join(IN_DIR, "results_scenario_4_bottleneck.csv")
    if not os.path.exists(file_path):
        print("Fișierul nu a fost găsit")
        return

    df = pd.read_csv(file_path)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", context="talk")

    sns.lineplot(
        data=df,
        x='tick',
        y='people_in_queue',
        errorbar=None,
        color='#c0392b',
        linewidth=3
    )

    plt.title("Evoluția cozii de așteptare pe scări în Căminul T17", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Timp (ticks)", fontsize=14)
    plt.ylabel("Studenți în așteptare", fontsize=14)

    max_active_tick = df[df['people_in_queue'] > 0]['tick'].max()
    if pd.isna(max_active_tick):
        x_limit = 1000
    else:
        x_limit = max_active_tick + 100

    plt.xlim(left=0, right=x_limit)
    plt.ylim(bottom=0)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Scenario_4_Bottleneck.png"), dpi=300, bbox_inches='tight')
    plt.close()


def firefighter_delay():
    file_path = os.path.join(IN_DIR, "results_scenario_5_firefighter_delay.csv")
    if not os.path.exists(file_path):
        print("Fișierul nu a fost găsit")
        return

    df = pd.read_csv(file_path)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7))

    palette = {'Fast': '#1f77b4', 'Normal': '#ff7f0e', 'Slow': '#d62728'}
    names = {'Fast': 'Rapid', 'Normal': 'Normal', 'Slow': 'Lent'}

    for mode in ['Fast', 'Normal', 'Slow']:
        mode_df = df[df['delay_mode'] == mode]

        last_fire_tick = mode_df[mode_df['fire_radius'] > 0]['tick'].max()
        mode_df_filtered = mode_df[mode_df['tick'] <= last_fire_tick + 100]

        mode_mean = mode_df_filtered.groupby('tick')['fire_radius'].mean().reset_index()
        mode_mean['fire_radius_smooth'] = mode_mean['fire_radius'].rolling(window=10, center=True, min_periods=1).mean()

        sns.lineplot(
            data=mode_mean,
            x='tick',
            y='fire_radius_smooth',
            label=names[mode],
            color=palette[mode],
            linewidth=3
        )

    plt.title("Evoluția focului în funcție de timpul de răspuns al pompierilor", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Timp (ticks)", fontsize=14)
    plt.ylabel("Raza focului (metri)", fontsize=14)
    plt.legend(title="Timpul de răspuns al pompierilor", loc='upper right', framealpha=1.0)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Scenario_5_Firefighter_Delay.png"), dpi=300, bbox_inches='tight')
    plt.close()

def campus_congestion():
    file_path = os.path.join(IN_DIR, "results_scenario_6_campus_congestion.csv")
    if not os.path.exists(file_path):
        print("Fișierul nu a fost găsit")
        return

    df = pd.read_csv(file_path)
    if df.empty:
        print("Datele sunt goale")
        return

    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    sns.set_theme(style="white", context="talk")

    plot = sns.kdeplot(
        data=df,
        x='x',
        y='y',
        fill=True,
        thresh=0.02,
        levels=25,
        cmap="turbo",
        alpha=0.7,
        ax=ax,
        zorder=2
    )
    if ax.collections:
        mappable = ax.collections[0]
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, label='Densitatea agenților')

    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs="EPSG:3857", zorder=1)
    except Exception as e:
        print(f"Nu s-a putut încărca harta de fundal. Eroare: {e}")

    plt.title("Heatmap al evacuării în campus", fontsize=20, fontweight='bold', pad=15)

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Scenario_6_Campus_Congestion.png"), dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":

    firefighter_delay()