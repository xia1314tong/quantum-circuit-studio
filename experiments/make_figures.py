"""由 ``experiments/results/*.csv`` 生成论文用图（300 dpi，PDF + PNG）。

用法::

    python -m experiments.make_figures

输出目录 ``experiments/figures/``：

* ``fig1_tvd_vs_p``      噪声定律：不同 n 下 TVD 随 p 的变化
* ``fig2_threshold``     教学可演示边界 p*(n) 与渐近线 eps/(2(n-1))
* ``fig3_mitigation``    四种缓解方案的对照（n=8，20000 shots）
* ``fig4_robustness``   模型失配下本文方法的稳健性
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(BASE, "figures")

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "none": "#4C4C4C",
    "readout": "#8C8C8C",
    "zne": "#E69F00",
    "model": "#0072B2",
    "model_est": "#56B4E9",
}
LABELS = {
    "none": "No mitigation",
    "readout": "Readout calibration",
    "zne": "ZNE (linear)",
    "model": "This work (known $p$)",
    "model_est": "This work (fitted $p$)",
}


def load(name: str):
    path = os.path.join(RESULTS, name)
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, stem: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            os.path.join(FIGURES, f"{stem}.{extension}"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


def fig1_noise_law() -> None:
    rows = load("noise_law.csv")
    series = defaultdict(lambda: ([], []))
    for row in rows:
        n = int(row["num_wires"])
        series[n][0].append(float(row["p"]))
        series[n][1].append(float(row["tvd"]))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, n in enumerate(sorted(series)):
        xs, ys = series[n]
        axes[0].plot(
            xs, ys, color=cmap(index / 8.0), label=rf"$n={n}$"
        )
    axes[0].set_xlabel(r"bit-flip probability $p$")
    axes[0].set_ylabel(r"TVD to ideal GHZ")
    axes[0].set_title("(a) noise law")
    axes[0].legend(ncol=2, frameon=False)
    axes[0].set_xlim(0, 0.5)
    axes[0].set_ylim(0, 1.0)

    # 双对数：p*(n) 幂律
    thresholds = load("threshold.csv")
    picks = defaultdict(lambda: ([], []))
    for row in thresholds:
        if abs(float(row["epsilon"]) - 0.10) > 1e-9:
            continue
        n = int(row["num_wires"])
        if n < 2:
            continue
        picks["data"][0].append(n)
        picks["data"][1].append(float(row["p_star"]))
        picks["asym"][0].append(n)
        picks["asym"][1].append(float(row["p_star_asymptotic"]))

    axes[1].loglog(
        picks["data"][0], picks["data"][1], "o-", color="#0072B2",
        markersize=4, label=r"$p^*$ (exact, $\varepsilon=0.1$)",
    )
    axes[1].loglog(
        picks["asym"][0], picks["asym"][1], "--", color="#E69F00",
        label=r"$\varepsilon/[2(n-1)]$",
    )
    axes[1].set_xlabel("qubits $n$")
    axes[1].set_ylabel(r"threshold $p^*$")
    axes[1].set_title(r"(b) teaching threshold")
    axes[1].legend(frameon=False)

    save(fig, "fig1_noise_law")


def fig2_threshold() -> None:
    thresholds = load("threshold.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
    styles = {0.05: ("s", "#009E73"), 0.10: ("o", "#0072B2")}
    for epsilon, (marker, color) in styles.items():
        xs, ys, asymptotes = [], [], []
        for row in thresholds:
            if abs(float(row["epsilon"]) - epsilon) > 1e-9:
                continue
            n = int(row["num_wires"])
            if n < 2:
                continue
            xs.append(n)
            ys.append(float(row["p_star"]))
            asymptotes.append(float(row["p_star_asymptotic"]))
        ax.loglog(xs, ys, marker + "-", color=color, markersize=4,
                  label=rf"$\varepsilon={epsilon:.2f}$ (exact)")
        ax.loglog(xs, asymptotes, ":", color=color, label=rf"$\varepsilon={epsilon:.2f}$ (asymptotic)")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"teaching threshold $p^*$")
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "fig2_threshold")


def fig3_mitigation() -> None:
    rows = load("mitigation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)
    for panel, n_target in enumerate((4, 8)):
        series = defaultdict(lambda: ([], []))
        for row in rows:
            if int(row["num_wires"]) != n_target:
                continue
            method = row["method"]
            series[method][0].append(float(row["p"]))
            series[method][1].append(float(row["tvd_mean"]))
        for method in ("none", "readout", "zne", "model", "model_est"):
            if method not in series:
                continue
            xs, ys = series[method]
            axes[panel].plot(
                xs, ys, color=COLORS[method], label=LABELS[method],
                linestyle="--" if method == "readout" else "-",
            )
        axes[panel].axhline(0.1, color="grey", linewidth=0.7, linestyle=":")
        axes[panel].set_xlim(0, 0.25)
        axes[panel].set_ylim(0, 0.9)
        axes[panel].set_xlabel(r"bit-flip probability $p$")
        axes[panel].set_title(rf"({'ab'[panel]}) $n={n_target}$, 20k shots")
        if panel == 0:
            axes[panel].set_ylabel("TVD after mitigation")
    axes[1].legend(frameon=False, loc="upper left", fontsize=6.5)
    save(fig, "fig3_mitigation")


def fig4_robustness() -> None:
    rows = load("robustness.csv")
    picks = defaultdict(list)
    for row in rows:
        if int(row["num_wires"]) != 6:
            continue
        picks[float(row["p"])].append(
            (float(row["model_error"]), float(row["tvd_model"]))
        )

    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
    cmap = plt.get_cmap("plasma")
    for index, p in enumerate(sorted(picks)):
        pairs = sorted(picks[p])
        xs = [pair[0] * 100 for pair in pairs]
        ys = [pair[1] for pair in pairs]
        ax.plot(xs, ys, "o-", markersize=3.5, color=cmap(index / 4.0),
                label=rf"$p={p:.2f}$")
    ax.set_xlabel(r"model error in $p$ (%)")
    ax.set_ylabel("TVD after inversion")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig4_robustness")


def fig5_llm_repair() -> None:
    """LLM 生成电路：一次性 vs 校验-反馈-修复。"""
    runs_path = os.path.join(RESULTS, "llm_runs.csv")
    if not os.path.exists(runs_path):
        print("[skip] llm_runs.csv 不存在")
        return
    with open(runs_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("[skip] llm_runs.csv 为空")
        return

    tasks = []
    for row in rows:
        if row["task"] not in tasks:
            tasks.append(row["task"])
    for row in rows:
        row["arm"] = row.get("arm") or "feedback"
    arms = [arm for arm in ("resample", "feedback") if any(row["arm"] == arm for row in rows)]

    def per_task(arm: str):
        success = []
        for task in tasks:
            task_rows = [row for row in rows if row["task"] == task and row["arm"] == arm]
            reps = sorted({row["rep"] for row in task_rows})
            if not reps:
                success.append(0.0)
                continue
            hit = 0
            for rep in reps:
                attempt = [row for row in task_rows if row["rep"] == rep]
                hit += int(any(row["passed"] == "1" for row in attempt))
            success.append(hit / len(reps))
        return success

    def first_attempt(arm: str):
        """第一轮就通过的比例（one-shot 口径）。"""
        success = []
        for task in tasks:
            task_rows = [
                row for row in rows if row["task"] == task and row["arm"] == arm and row["round"] == "1"
            ]
            if not task_rows:
                success.append(0.0)
                continue
            success.append(sum(int(row["passed"] == "1") for row in task_rows) / len(task_rows))
        return success

    one_shot = first_attempt(arms[-1])
    curves = [(arm, per_task(arm)) for arm in arms]

    layers = {"syntax": 0, "semantic": 0, "numeric": 0, "truncated": 0, "api_error": 0}
    layer_by_arm = {arm: {key: 0 for key in layers} for arm in arms}
    for row in rows:
        if row["passed"] != "1" and row["layer"] in layers:
            layers[row["layer"]] += 1
            layer_by_arm.setdefault(row["arm"], {key: 0 for key in layers})
            layer_by_arm[row["arm"]][row["layer"]] = layer_by_arm[row["arm"]].get(row["layer"], 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2.6, 1.0]})
    xs = np.arange(len(tasks))
    width = 0.8 / (len(curves) + 1)
    axes[0].bar(xs - width, one_shot, width, label="one-shot", color="#8C8C8C")
    colors = {"resample": "#E8A33D", "feedback": "#2F6DB3", "oneshot": "#8C8C8C"}
    for index, (arm, values) in enumerate(curves):
        axes[0].bar(
            xs + width * (index - 0.5) + width * 0.5,
            values,
            width,
            label=f"{arm} (k=3)",
            color=colors.get(arm, "#2F6DB3"),
        )
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(tasks, rotation=40, ha="right", fontsize=6)
    axes[0].set_ylim(0, 1.3)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("(a) per task")
    axes[0].legend(frameon=False, loc="upper center", ncols=3, fontsize=6.5)

    labels = list(layers)
    xs2 = np.arange(len(labels))
    for index, arm in enumerate(arms):
        values = [layer_by_arm.get(arm, {}).get(key, 0) for key in labels]
        axes[1].bar(xs2 + (index - 0.5) * 0.38, values, 0.38, label=arm,
                    color=colors.get(arm, "#2F6DB3"))
    axes[1].set_xticks(xs2)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
    axes[1].set_ylabel("failed attempts")
    axes[1].set_title("(b) failure layer")
    save(fig, "fig5_llm_repair")


def fig6_cross_model() -> None:
    """跨模型对照：不同模型下 resample / feedback 的最终成功率与配对差。"""
    path = os.path.join(RESULTS, "llm_cross_model.csv")
    if not os.path.exists(path):
        print("[skip] fig6_cross_model：还没有 llm_cross_model.csv")
        return
    rows = load("llm_cross_model.csv")
    models = sorted({row["model"] for row in rows})
    arms = [arm for arm in ("resample", "feedback") if any(row["arm"] == arm for row in rows)]
    lookup = {(row["model"], row["arm"]): row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.5, 1.0]})
    palette = {"resample": "#E8A33D", "feedback": "#2F6DB3"}
    xs = np.arange(len(models))
    width = 0.8 / max(len(arms), 1)
    matched = True
    for index, arm in enumerate(arms):
        values = []
        for model in models:
            pair_row = lookup.get((model, "feedback"), {})
            key = "rate_resample" if arm == "resample" else "rate_feedback"
            raw = pair_row.get(key, "")
            if raw in ("", None):
                matched = False
                values.append(float(lookup[(model, arm)]["final_pass"]))
            else:
                values.append(float(raw))
        axes[0].bar(xs + (index - (len(arms) - 1) / 2) * width, values, width,
                    label=arm, color=palette.get(arm, "#8C8C8C"))
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([model.replace("deepseek-", "") for model in models], fontsize=7)
    axes[0].set_ylim(0.6, 1.03)
    axes[0].set_ylabel("success rate (matched pairs)" if matched else "final success rate")
    axes[0].set_title("(a) success on matched pairs" if matched else "(a) success after k=3")
    axes[0].legend(frameon=False, fontsize=7)

    deltas, lows, highs, pvalues = [], [], [], []
    for model in models:
        row = lookup.get((model, "feedback"), {})
        delta = float(row.get("delta_vs_resample") or 0.0)
        low = float(row.get("ci_low") or 0.0)
        high = float(row.get("ci_high") or 0.0)
        deltas.append(delta * 100)
        lows.append((delta - low) * 100)
        highs.append((high - delta) * 100)
        pvalues.append(row.get("mcnemar_p", ""))
    axes[1].bar(xs, deltas, 0.45, color="#2F6DB3")
    axes[1].errorbar(xs, deltas, yerr=[lows, highs], fmt="none", ecolor="#333333",
                     elinewidth=0.9, capsize=3)
    for index, value in enumerate(pvalues):
        try:
            text = f"p={float(value):.2f}"
        except (TypeError, ValueError):
            text = ""
        if text:
            axes[1].annotate(text, (xs[index], deltas[index]), textcoords="offset points",
                             xytext=(0, 8 if deltas[index] >= 0 else -14),
                             ha="center", fontsize=6.5)
    axes[1].axhline(0, color="#333333", linewidth=0.7)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([model.replace("deepseek-", "") for model in models], fontsize=7)
    axes[1].set_ylabel("$\\Delta$ success (pp)")
    axes[1].set_title("(b) feedback $-$ resample")
    save(fig, "fig6_cross_model")


def main() -> None:
    fig1_noise_law()
    fig2_threshold()
    fig3_mitigation()
    fig4_robustness()
    fig5_llm_repair()
    fig6_cross_model()
    for name in sorted(os.listdir(FIGURES)):
        print(f"[ok] experiments/figures/{name}")


if __name__ == "__main__":
    main()
