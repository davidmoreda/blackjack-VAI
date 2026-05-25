"""Genera graficas de la presentacion desde mlflow.db consolidado."""
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "presentation_assets"
OUT.mkdir(exist_ok=True, parents=True)

con = sqlite3.connect(str(ROOT / "mlflow.db"))
cur = con.cursor()

plt.style.use("dark_background")
COLORS = {
    "V3":      "#6b7280",   # gris medio
    "V4":      "#60a5fa",   # azul
    "V5":      "#a78bfa",   # lila
    "V6":      "#D4A017",   # dorado: modelo de produccion
    "RT-DETR": "#f87171",   # rojo coral
}

RUNS = [
    ("V3",      "23ed873260c2417c973d0f32cfbc2a2b", "yolo"),
    ("V4",      "0fca74dda1394412b869932918e5a4b0", "yolo"),
    ("V5",      "5873e95c98ec4399a69d218190224a37", "yolo"),
    ("V6",      "c661dbcb46494396a932e1a426165d5a", "yolo"),
    ("RT-DETR", None, "rtdetr"),
]

rt_row = cur.execute(
    "SELECT r.run_uuid FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id "
    "WHERE e.name = 'blackjackvai-rtdetr-matched' AND r.name = 'rtdetr_l_matched'"
).fetchone()
if rt_row:
    RUNS[-1] = ("RT-DETR", rt_row[0], "rtdetr")
    print("RT-DETR run:", rt_row[0])

def get_history(run_id, possible_keys):
    for key in possible_keys:
        rows = list(cur.execute(
            "SELECT step, value FROM metrics WHERE run_uuid = ? AND key = ? ORDER BY step",
            (run_id, key)))
        if rows:
            seen = {}
            for s, v in rows:
                seen[s] = v
            steps = sorted(seen.keys())
            return steps, [seen[s] for s in steps]
    return [], []

KEYS = {
    "map50":   {"yolo": ["metrics/mAP50B"], "rtdetr": ["metrics/mAP50B"]},
    "map5095": {"yolo": ["metrics/mAP50-95B"], "rtdetr": ["metrics/mAP50-95B"]},
    "p":       {"yolo": ["metrics/precisionB"], "rtdetr": ["metrics/precisionB"]},
    "r":       {"yolo": ["metrics/recallB"], "rtdetr": ["metrics/recallB"]},
}

# ---------- 1) Evolucion de modelos ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, run_id, backend in RUNS:
    if not run_id:
        continue
    s, v = get_history(run_id, KEYS["map50"][backend])
    if s:
        lw = 3 if label == "V6" else 1.8
        axes[0].plot(s, v, label=label, color=COLORS[label], linewidth=lw)
    s, v = get_history(run_id, KEYS["map5095"][backend])
    if s:
        lw = 3 if label == "V6" else 1.8
        axes[1].plot(s, v, label=label, color=COLORS[label], linewidth=lw)

for ax, title in [(axes[0], "mAP@50 por epoch"),
                   (axes[1], "mAP@50-95 por epoch")]:
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=10)

plt.suptitle("Evolucion de modelos",
             fontsize=14, fontweight="bold", color="#D4A017")
plt.tight_layout()
plt.savefig(OUT / "01_evolucion_modelos.png", dpi=160, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print("01_evolucion_modelos.png")

# ---------- 2) Metricas finales (barras) ----------
labels, m50, m5095, prec, rec = [], [], [], [], []
for lbl, run_id, backend in RUNS:
    if not run_id:
        continue
    labels.append(lbl)
    _, v = get_history(run_id, KEYS["map50"][backend]);   m50.append(max(v) if v else 0)
    _, v = get_history(run_id, KEYS["map5095"][backend]); m5095.append(max(v) if v else 0)
    _, v = get_history(run_id, KEYS["p"][backend]);       prec.append(max(v) if v else 0)
    _, v = get_history(run_id, KEYS["r"][backend]);       rec.append(max(v) if v else 0)

x = np.arange(len(labels))
w = 0.2
fig, ax = plt.subplots(figsize=(11, 5))
groups = [
    (ax.bar(x - 1.5*w, m50,   w, label="mAP@50",    color="#60a5fa")),
    (ax.bar(x - 0.5*w, m5095, w, label="mAP@50-95", color="#D4A017")),
    (ax.bar(x + 0.5*w, prec,  w, label="Precision", color="#4ade80")),
    (ax.bar(x + 1.5*w, rec,   w, label="Recall",    color="#a78bfa")),
]
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.85, 1.0)
ax.set_ylabel("Metrica (split val)")
ax.set_title("Metricas finales por modelo - mejor epoch",
              fontsize=14, fontweight="bold", color="#D4A017")
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="lower left", fontsize=10)
for g in groups:
    for b in g:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h),
                     ha="center", va="bottom", fontsize=7, alpha=0.85)
plt.tight_layout()
plt.savefig(OUT / "02_metricas_finales.png", dpi=160, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print("02_metricas_finales.png")

# ---------- 3) Curvas detalladas del modelo de produccion (V6) ----------
V5_ID = "c661dbcb46494396a932e1a426165d5a"  # V6 run id (mantengo variable name por compatibilidad)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
panels = [
    (KEYS["map50"]["yolo"],   "mAP@50"),
    (KEYS["map5095"]["yolo"], "mAP@50-95"),
    (KEYS["p"]["yolo"],       "Precision"),
    (KEYS["r"]["yolo"],       "Recall"),
]
for ax, (key_opts, title) in zip(axes.flatten(), panels):
    s, v = get_history(V5_ID, key_opts)
    ax.plot(s, v, color="#D4A017", linewidth=2.5)
    ax.fill_between(s, v, alpha=0.15, color="#D4A017")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    if v:
        peak = max(v)
        peak_epoch = s[v.index(peak)]
        ax.scatter([peak_epoch], [peak], color="#fff", zorder=5, s=80,
                     edgecolors="#D4A017", linewidths=2)
        ax.annotate(f"max={peak:.4f}\n(epoch {peak_epoch})",
                     (peak_epoch, peak), textcoords="offset points",
                     xytext=(10, -15), fontsize=9, color="#fff")
plt.suptitle("Modelo de produccion - YOLOv8m-seg V6 (curvas de entrenamiento)",
             fontsize=14, fontweight="bold", color="#D4A017")
plt.tight_layout()
plt.savefig(OUT / "03_modelo_produccion_v6.png", dpi=160, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print("03_modelo_produccion_v6.png")

# ---------- 5) V6: Train vs Val losses ----------
loss_panels = [
    ("box_loss", "Box Loss",       "#60a5fa"),
    ("cls_loss", "Classification", "#D4A017"),
    ("dfl_loss", "DFL Loss",       "#4ade80"),
    ("seg_loss", "Segmentation",   "#a78bfa"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (key, title, color) in zip(axes.flatten(), loss_panels):
    st, vt = get_history(V5_ID, [f"train/{key}"])
    sv, vv = get_history(V5_ID, [f"val/{key}"])
    if st:
        ax.plot(st, vt, label="train", color=color, linewidth=2.2)
    if sv:
        ax.plot(sv, vv, label="val", color=color, linewidth=2.2,
                linestyle="--", alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
plt.suptitle("YOLOv8m-seg V6 - Train vs Val losses (sin overfitting)",
             fontsize=14, fontweight="bold", color="#D4A017")
plt.tight_layout()
plt.savefig(OUT / "05_v6_losses.png", dpi=160, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print("05_v6_losses.png")

# ---------- 6) V6: Box vs Mask mAP ----------
fig, ax = plt.subplots(figsize=(11, 5.5))
curves = [
    ("metrics/mAP50B",    "Box mAP@50",     "#60a5fa", 2.5, "-"),
    ("metrics/mAP50-95B", "Box mAP@50-95",  "#D4A017", 2.5, "-"),
    ("metrics/mAP50M",    "Mask mAP@50",    "#4ade80", 2.5, "--"),
    ("metrics/mAP50-95M", "Mask mAP@50-95", "#a78bfa", 2.5, "--"),
]
for key, label, color, lw, ls in curves:
    s, v = get_history(V5_ID, [key])
    if s:
        ax.plot(s, v, label=label, color=color, linewidth=lw, linestyle=ls)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("mAP", fontsize=11)
ax.set_ylim(0.85, 1.0)
ax.set_title("YOLOv8m-seg V6 - mAP de deteccion vs segmentacion",
             fontsize=14, fontweight="bold", color="#D4A017")
ax.grid(alpha=0.2)
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "06_v6_box_vs_mask.png", dpi=160, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print("06_v6_box_vs_mask.png")

# ---------- 9) V6: results.png IDENTICO a ultralytics (2x9 fondo claro) ----------
with plt.style.context("default"):
    panels_official = [
        # Row 0
        ("train/box_loss", "train/box_loss"),
        ("train/seg_loss", "train/seg_loss"),
        ("train/cls_loss", "train/cls_loss"),
        ("train/dfl_loss", "train/dfl_loss"),
        ("train/sem_loss", "train/sem_loss"),
        ("metrics/precisionB", "metrics/precision(B)"),
        ("metrics/recallB",    "metrics/recall(B)"),
        ("metrics/mAP50B",     "metrics/mAP50(B)"),
        ("metrics/mAP50-95B",  "metrics/mAP50-95(B)"),
        # Row 1
        ("val/box_loss", "val/box_loss"),
        ("val/seg_loss", "val/seg_loss"),
        ("val/cls_loss", "val/cls_loss"),
        ("val/dfl_loss", "val/dfl_loss"),
        ("val/sem_loss", "val/sem_loss"),
        ("metrics/precisionM", "metrics/precision(M)"),
        ("metrics/recallM",    "metrics/recall(M)"),
        ("metrics/mAP50M",     "metrics/mAP50(M)"),
        ("metrics/mAP50-95M",  "metrics/mAP50-95(M)"),
    ]
    fig, axes = plt.subplots(2, 9, figsize=(22, 6))
    for i, (key, title) in enumerate(panels_official):
        ax = axes[i // 9, i % 9]
        s, v = get_history(V5_ID, [key])
        if s:
            ax.plot(s, v, "-o", color="#1f77b4", markersize=2.5,
                     linewidth=1.2, label="results")
            # Smooth (rolling mean window=5, centered, sin artifacts de borde)
            if len(v) >= 5:
                v_smooth = pd.Series(v).rolling(window=5, center=True,
                                                 min_periods=1).mean().values
                ax.plot(s, v_smooth, "--", color="#ff7f0e",
                         linewidth=1.5, label="smooth")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "09_v6_results_official.png", dpi=160,
                 bbox_inches="tight", facecolor="white")
    plt.close()
    print("09_v6_results_official.png")

# ---------- 8) V6: results.png estilo casino (2x6 con todo) ----------
panels = [
    # Row 0 - train losses
    ("train/box_loss",     "train/box_loss",     "#60a5fa"),
    ("train/cls_loss",     "train/cls_loss",     "#D4A017"),
    ("train/dfl_loss",     "train/dfl_loss",     "#4ade80"),
    ("train/seg_loss",     "train/seg_loss",     "#a78bfa"),
    ("metrics/precisionB", "metrics/precision",  "#f87171"),
    ("metrics/mAP50B",     "metrics/mAP50",      "#D4A017"),
    # Row 1 - val losses + recall + map50-95
    ("val/box_loss",       "val/box_loss",       "#60a5fa"),
    ("val/cls_loss",       "val/cls_loss",       "#D4A017"),
    ("val/dfl_loss",       "val/dfl_loss",       "#4ade80"),
    ("val/seg_loss",       "val/seg_loss",       "#a78bfa"),
    ("metrics/recallB",    "metrics/recall",     "#f87171"),
    ("metrics/mAP50-95B",  "metrics/mAP50-95",   "#D4A017"),
]
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
for i, (key, title, color) in enumerate(panels):
    ax = axes[i // 6, i % 6]
    s, v = get_history(V5_ID, [key])
    if s:
        ax.plot(s, v, color=color, linewidth=2)
        ax.fill_between(s, v, alpha=0.12, color=color)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("epoch", fontsize=9)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", labelsize=8)
plt.suptitle("YOLOv8m-seg V6 - Resumen completo de entrenamiento (losses + metricas)",
             fontsize=14, fontweight="bold", color="#D4A017", y=1.00)
plt.tight_layout()
plt.savefig(OUT / "08_v6_results_summary.png", dpi=160, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print("08_v6_results_summary.png")

# ---------- 7) V6: Learning rate schedule ----------
fig, ax = plt.subplots(figsize=(11, 4))
for pg, color in [("lr/pg0", "#D4A017"), ("lr/pg1", "#60a5fa"), ("lr/pg2", "#a78bfa")]:
    s, v = get_history(V5_ID, [pg])
    if s:
        ax.plot(s, v, label=pg, color=color, linewidth=2)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Learning rate", fontsize=11)
ax.set_title("YOLOv8m-seg V6 - Learning rate schedule (cos_lr, lr0=0.01)",
             fontsize=14, fontweight="bold", color="#D4A017")
ax.grid(alpha=0.2)
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "07_v6_lr_schedule.png", dpi=160, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print("07_v6_lr_schedule.png")

# ---------- 4) Coste e inferencia (de comparison_runs/comparison_summary.csv) ----------
csv_path = ROOT / "comparison_runs" / "comparison_summary.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path).set_index("model")
    colors = ["#D4A017", "#f87171"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(df.index, df["fps"], color=colors)
    axes[0].set_title("FPS efectivo @ imgsz=640", fontweight="bold")
    axes[0].set_ylabel("FPS"); axes[0].grid(axis="y", alpha=0.2)
    for i, v in enumerate(df["fps"]):
        axes[0].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")

    axes[1].bar(df.index, df["latency_ms_mean"], color=colors)
    axes[1].set_title("Latencia media (ms/img)", fontweight="bold")
    axes[1].set_ylabel("ms"); axes[1].grid(axis="y", alpha=0.2)
    for i, v in enumerate(df["latency_ms_mean"]):
        axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")

    axes[2].bar(df.index, df["params_M"], color=colors)
    axes[2].set_title("Parametros (M)", fontweight="bold")
    axes[2].set_ylabel("M params"); axes[2].grid(axis="y", alpha=0.2)
    for i, v in enumerate(df["params_M"]):
        axes[2].text(i, v + 0.3, f"{v:.1f}M", ha="center", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", rotation=10)

    plt.suptitle("Comparativa de coste e inferencia - YOLO V6 vs RT-DETR matched",
                 fontsize=13, fontweight="bold", color="#D4A017")
    plt.tight_layout()
    plt.savefig(OUT / "04_comparativa_inferencia.png", dpi=160, bbox_inches="tight",
                 facecolor="#0e1117")
    plt.close()
    print("04_comparativa_inferencia.png")
else:
    print("AVISO: no encontre", csv_path)

# ---------- 10) Losses comparativos - hold-on de los 5 modelos (V6 resaltado) ----------
# YOLO loss schema: train/{box,cls,dfl,seg}_loss + val/*
# RT-DETR loss schema: train/{cls,giou,l1}_loss + val/*
# -> RT-DETR solo se superpone en el panel "Classification" (cls_loss),
#    en el resto se anota "no aplica" (arquitectura distinta).
loss_panels_compare = [
    ("box_loss", "Box Loss",       True),
    ("cls_loss", "Classification", True),   # RT-DETR si aplica aqui
    ("dfl_loss", "DFL Loss",       False),
    ("seg_loss", "Segmentation",   False),
]
for split in ("train", "val"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, title, rtdetr_ok) in zip(axes.flatten(), loss_panels_compare):
        for label, run_id, backend in RUNS:
            if not run_id:
                continue
            if backend == "rtdetr" and not rtdetr_ok:
                continue
            s, v = get_history(run_id, [f"{split}/{key}"])
            if not s:
                continue
            is_v6 = (label == "V6")
            ax.plot(
                s, v,
                label=label,
                color=COLORS[label],
                linewidth=3.2 if is_v6 else 1.6,
                alpha=1.0 if is_v6 else 0.85,
                zorder=5 if is_v6 else 2,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{split} loss")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="upper right")
        if not rtdetr_ok:
            ax.text(
                0.98, 0.02, "RT-DETR: no aplica\n(arquitectura distinta)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#9ca3af", alpha=0.85,
            )
    plt.suptitle(
        f"Comparativa de {split} losses por modelo (V6 resaltado)",
        fontsize=14, fontweight="bold", color="#D4A017",
    )
    plt.tight_layout()
    out_path = OUT / f"10_compare_{split}_losses.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(out_path.name)

# ---------- 11) mAP comparativos - hold-on de los 5 modelos (V6 resaltado) ----------
# RT-DETR es detector puro -> sin mask mAP. En los paneles Mask se anota "no aplica".
map_panels_compare = [
    ("metrics/mAP50B",    "Box mAP@50",     True),
    ("metrics/mAP50-95B", "Box mAP@50-95",  True),
    ("metrics/mAP50M",    "Mask mAP@50",    False),
    ("metrics/mAP50-95M", "Mask mAP@50-95", False),
]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, (key, title, rtdetr_ok) in zip(axes.flatten(), map_panels_compare):
    for label, run_id, backend in RUNS:
        if not run_id:
            continue
        if backend == "rtdetr" and not rtdetr_ok:
            continue
        s, v = get_history(run_id, [key])
        if not s:
            continue
        is_v6 = (label == "V6")
        ax.plot(
            s, v,
            label=label,
            color=COLORS[label],
            linewidth=3.2 if is_v6 else 1.6,
            alpha=1.0 if is_v6 else 0.85,
            zorder=5 if is_v6 else 2,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0.80, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")
    if not rtdetr_ok:
        ax.text(
            0.98, 0.02, "RT-DETR: solo deteccion\n(sin mask mAP)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#9ca3af", alpha=0.85,
        )
plt.suptitle(
    "Comparativa de mAP por modelo (V6 resaltado)",
    fontsize=14, fontweight="bold", color="#D4A017",
)
plt.tight_layout()
plt.savefig(OUT / "11_compare_mAP.png", dpi=160, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print("11_compare_mAP.png")

print("\nDone. Output:", OUT)
