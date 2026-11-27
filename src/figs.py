"""figs.py - style và hàm vẽ Fig. 1-4 (vẽ ở kích thước in thật của IEEE)."""
import numpy as np
import matplotlib.pyplot as plt

COL1, COL2 = 3.5, 7.16 # inch: 1 cột / 2 cột IEEE
C = dict(spam="#D55E00", ham="#0072B2", prev="#009E73", inten="#CC79A7")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none",
})

def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold")
    
def fig1_class_distribution(n_raw, n_dedup):
    """n_raw, n_dedup: dict {'ham': int, 'spam': int}."""
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    x = np.arange(2); w = 0.36
    for i, (lab, col) in enumerate((("ham", C["ham"]), ("spam", C["spam"]))):
        vals = [n_raw[lab], n_dedup[lab]]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=lab.capitalize())
        tot = [sum(n_raw.values()), sum(n_dedup.values())]
        for b, v, t in zip(bars, vals, tot):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / t:.1%})",
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, ["Original (n = %s)" % f"{sum(n_raw.values()):,}",
                      "Deduplicated (n = %s)" % f"{sum(n_dedup.values()):,}"])
    ax.set_ylabel("Number of messages")
    ax.set_ylim(0, max(n_raw.values()) * 1.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

def fig2_top_terms(top):
    """top: bảng tidy 40 dòng. Hai panel riêng, trục riêng."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.2))
    for ax, side, lab in zip(axes, ("Spam", "Ham"), "ab"):
        d = top[top.side == side].iloc[::-1] # hạng 1 nằm trên cùng
        y = np.arange(len(d)); h = 0.38
        ax.barh(y + h / 2, d.m_S, h, color=C["spam"], label="Spam")
        ax.barh(y - h / 2, d.m_H, h, color=C["ham"], label="Ham")
        ax.set_yticks(y, d.term); ax.set_xlabel("Mean TF-IDF (unitless)")
        rank = "Δ" if side == "Spam" else "−Δ"
        ax.set_title(f"{side}-side terms (ranked by {rank})", loc="left")
        panel_label(ax, lab)
    axes[0].legend(frameon=False, loc="lower right"); fig.tight_layout()
    return fig

def stacked_signed(ax, y, comps, colors, labels, h=0.7):
    """Xếp chồng Δprev + Δint, đúng cả khi hai thành phần trái dấu."""
    pos, neg = np.zeros(len(y)), np.zeros(len(y))
    for v, c, l in zip(comps, colors, labels):
        v = np.asarray(v, float)
        ax.barh(y, v, left=np.where(v >= 0, pos, neg), color=c, label=l, height=h)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
        
def fig3_decomposition(top, lengths_S, lengths_H):
    fig = plt.figure(figsize=(COL2, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
    for k, (side, lab) in enumerate((("Spam", "a"), ("Ham", "b"))):
        ax = fig.add_subplot(gs[k]); d = top[top.side == side].iloc[::-1]
        y = np.arange(len(d))
        stacked_signed(ax, y, [d.d_prev, d.d_int], [C["prev"], C["inten"]],
                       ["Δ prevalence", "Δ intensity"])
        ax.axvline(0, color="k", lw=0.5); ax.set_yticks(y, d.term)
        ax.set_xlabel("Δ mean TF-IDF (unitless)"); panel_label(ax, lab)
        if k == 0: ax.legend(frameon=False)
    ax = fig.add_subplot(gs[2]); bins = np.linspace(0, 400, 41)
    ax.hist(lengths_H, bins, density=True, alpha=.7, color=C["ham"], label="Ham")
    ax.hist(lengths_S, bins, density=True, alpha=.7, color=C["spam"], label="Spam")
    ax.set_xlabel("Message length (characters)"); ax.set_ylabel("Density")
    ax.legend(frameon=False); panel_label(ax, "c"); fig.tight_layout()
    return fig

def fig4_jaccard(mats, titles):
    """mats: list 4 DataFrame 4x4 (Spam-all, Ham-all, Spam-uni, Ham-uni)."""
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.0))
    for ax, M, t, lab in zip(axes, mats, titles, "abcd"):
        im = ax.imshow(M.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(4), M.columns); ax.set_yticks(range(4), M.index)
        for i in range(4):
            for j in range(4):
                v = M.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
        ax.set_title(t, fontsize=6.5); panel_label(ax, lab)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Jaccard index")
    return fig