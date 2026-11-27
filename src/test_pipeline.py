"""test_pipeline.py - 5 kiểm tra tự động (chạy: pytest -q test_pipeline.py)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from tfidf_lib import TOKEN_MAIN, compute_table, make_configs

def test_toy_hand_calculation(): # Kiểm tra 1
    docs = ["free call now", "call me now", "see you"]
    vec = TfidfVectorizer(token_pattern=TOKEN_MAIN, sublinear_tf=True, min_df=1)
    X = vec.fit_transform(docs).toarray()
    n = 3
    idf_free = np.log((1 + n) / (1 + 1)) + 1 # 'free' xuất hiện trong 1 tin
    idf_call = np.log((1 + n) / (1 + 2)) + 1 # 'call' xuất hiện trong 2 tin
    raw = np.array([idf_free, idf_call, idf_call]) # free, call, now; tf = 1
    expected = raw / np.linalg.norm(raw) # chuẩn hóa L2
    got = [X[0, vec.vocabulary_[t]] for t in ("free", "call", "now")]
    assert np.allclose(got, expected)
    
def test_rows_have_unit_l2_norm(): # Kiểm tra 2
    docs = ["win a free prize now", "ok see you at 5", "call now to claim"]
    X = TfidfVectorizer(token_pattern=TOKEN_MAIN, sublinear_tf=True).fit_transform(docs)
    assert np.allclose(np.sqrt(X.multiply(X).sum(axis=1)).A1, 1.0)
    
def test_identities_on_real_table(df): # Kiểm tra 3 (assert nằm sẵn trong compute_table)
    tab, *_ = compute_table(df, make_configs(stopwords.words("english"))["C2"])
    assert (abs(tab.m_S - tab.DF_S * tab.c_S) < 1e-12).all()
    assert (abs(tab.delta - tab.d_prev - tab.d_int) < 1e-12).all()
    
def test_tokenizer_keeps_symbols(): # Kiểm tra 4
    an = TfidfVectorizer(token_pattern=TOKEN_MAIN).build_analyzer()
    toks = an("Win £1000 now!! txt me")
    for t in ("£", "1000", "!", "txt"):
        assert t in toks
        
def test_call_stopword_trap(): # Kiểm tra 5
    assert "call" not in stopwords.words("english")
    assert "call" in ENGLISH_STOP_WORDS # bẫy của sklearn