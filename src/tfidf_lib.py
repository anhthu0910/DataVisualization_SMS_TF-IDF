"""tfidf_lib.py - mọi phân tích đi qua run_analysis() (Research Plan v2, mục 3-4)."""

import re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

SEED = 42 # mọi bước ngẫu nhiên
EPS = 0.001 # epsilon cho LPR (cố định)
DF_MIN = 0.02 # ngưỡng DF bên trong chính lớp
N_TOP = 20 # số term mỗi phía
TOKEN_MAIN = r"(?u)\b\w[\w']*\b|[£$€!?]" # giữ £ $ € ! ? số, i'm
TOKEN_PLAIN = r"(?u)\b\w[\w']*\b" # C3/C4: bỏ £ ! ?

def make_configs(stopwords):
    """4 cấu hình tiền xử lý. stopwords = danh sách NLTK (không dùng của sklearn)."""
    return {
    "C1": dict(token_pattern=TOKEN_MAIN, stop_words=None, ngram_range=(1, 1)),
    "C2": dict(token_pattern=TOKEN_MAIN, stop_words=None, ngram_range=(1, 2)),
    "C3": dict(token_pattern=TOKEN_PLAIN, stop_words=stopwords, ngram_range=(1, 1)),
    "C4": dict(token_pattern=TOKEN_PLAIN, stop_words=stopwords, ngram_range=(1, 2)),
    }
    
def _col(M, axis=0):
    return np.asarray(M.sum(axis=axis)).ravel()

def compute_table(df, cfg, min_df=5):
    """Fit TF-IDF trên TOÀN corpus, rồi tính thống kê theo lớp cho MỌI term."""
    vec = TfidfVectorizer(lowercase=True, min_df=min_df, sublinear_tf=True,
    norm="l2", **cfg)
    X = vec.fit_transform(df["text"]).tocsr()
    y = (df["label"] == "spam").to_numpy()
    XS, XH = X[y], X[~y]
    nS, nH = XS.shape[0], XH.shape[0]
    kS, kH = _col(XS > 0), _col(XH > 0) # số tin chứa term
    sS, sH = _col(XS), _col(XH) # tổng x(t,d)
    pS, pH = kS / nS, kH / nH # DF theo lớp
    mS, mH = sS / nS, sH / nH # Mean TF-IDF (tính cả tin x = 0)
    
    cS = np.divide(sS, kS, out=np.zeros_like(sS), where=kS > 0) # Conditional Mean
    cH = np.divide(sH, kH, out=np.zeros_like(sH), where=kH > 0)
    delta = mS - mH
    d_prev = (pS - pH) * (cS + cH) / 2
    d_int = (cS - cH) * (pS + pH) / 2
    assert np.abs(mS - pS * cS).max() < 1e-12, "đồng nhất thức m = p * c bị sai"
    assert np.abs(mH - pH * cH).max() < 1e-12
    assert np.abs(delta - (d_prev + d_int)).max() < 1e-12, "phân rã Δ bị sai"
    tab = pd.DataFrame({
    "term": vec.get_feature_names_out(), "idf": vec.idf_,
    "k_S": kS, "k_H": kH, "DF_S": pS, "DF_H": pH,
    "m_S": mS, "m_H": mH, "c_S": cS, "c_H": cH,
    "delta": delta, "d_prev": d_prev, "d_int": d_int,
    "D": pS - pH, "LPR": np.log((pS + EPS) / (pH + EPS)),
    })
    return tab, vec, XS, XH

def _view_mask(terms, view):
    s = pd.Series(terms)
    uni = ~s.str.contains(" ")
    if view == "all":
        return np.ones(len(s), dtype=bool)
    if view == "unigram":
        return uni.to_numpy()
    if view == "alpha": # unigram chỉ gồm chữ cái (cho phép dấu ')
        return (uni & s.str.fullmatch(r"[a-z][a-z']*")).to_numpy()
    raise ValueError(view)

def select_top(tab, view="all", n=N_TOP, df_min=DF_MIN):
    """Bước 1: lọc DF >= 2% trong chính lớp. Bước 2: xếp hạng theo delta."""
    ok = tab[_view_mask(tab["term"], view)]
    S = (ok[ok["DF_S"] >= df_min]
        .sort_values(["delta", "term"], ascending=[False, True]).head(n)
        .assign(side="Spam"))
    H = (ok[ok["DF_H"] >= df_min]
        .sort_values(["delta", "term"], ascending=[True, True]).head(n)
        .assign(side="Ham"))
    return pd.concat([S, H], ignore_index=True)

def add_inference(top, vec, XS, XH):
    """Mann-Whitney U hai phía, Holm trên 40 phép thử, rank-biserial, Wilson CI."""
    nS, nH = XS.shape[0], XH.shape[0]
    XS, XH = XS.tocsc(), XH.tocsc()
    rows = []
    for t in top["term"]:
        j = vec.vocabulary_[t]
        xs, xh = XS[:, j].toarray().ravel(), XH[:, j].toarray().ravel()
        U, p = mannwhitneyu(xs, xh, alternative="two-sided", method="asymptotic")
        q = lambda a: np.percentile(a, [25, 75])
        rows.append(dict(U=U, p_raw=p, r=2 * U / (nS * nH) - 1,
            med_S=np.median(xs), q1_S=q(xs)[0], q3_S=q(xs)[1],
            med_H=np.median(xh), q1_H=q(xh)[0], q3_H=q(xh)[1]))
    out = top.reset_index(drop=True).join(pd.DataFrame(rows))
    out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    for g, n in (("S", nS), ("H", nH)):
        lo, hi = proportion_confint(out[f"k_{g}"], n, alpha=0.05, method="wilson")
        out[f"ci_lo_{g}"], out[f"ci_hi_{g}"] = lo, hi
    return out

def run_analysis(df, cfg, view="all", inference=False, min_df=5, return_full=False):
    """Hàm duy nhất: df (cột label, text) + cấu hình -> bảng tidy 40 term."""
    tab, vec, XS, XH = compute_table(df, cfg, min_df)
    top = select_top(tab, view)
    if inference:
        top = add_inference(top, vec, XS, XH)
    return (top, tab) if return_full else top

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan

def jaccard_matrix(sets):
    """sets: dict tên_cấu_hình -> list term. Trả về DataFrame 4x4."""
    names = list(sets)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            M.loc[a, b] = jaccard(sets[a], sets[b])
    return M

def show_examples(df, term, label, k=10, seed=SEED):
    """Đọc tin thật chứa term (kiểm tra ngữ cảnh)."""
    mask = df["text"].str.lower().str.contains(re.escape(term), regex=True)
    sub = df[(df["label"] == label) & mask]
    return sub.sample(min(k, len(sub)), random_state=seed)[["text"]]