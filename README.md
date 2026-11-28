# A Robustness-Checked TF-IDF Analysis of Distinctive Vocabulary in Spam and Legitimate SMS Messages

Final Individual Project · Data Visualisation · Faculty of Data Science and Artificial Intelligence, NEU

- **Tác giả:** Lê Ngọc Anh Thư - 11247355
- **Giảng viên:** ThS. Ngô Thái Hòa
- **Phiên bản kế hoạch:** Research Plan v2 (chốt ngày 26/11/2026)
- **Từ khóa:** TF-IDF, SMS spam, text analysis, distinctive terms, robustness, data visualisation

---

## 1. Mục tiêu và câu hỏi nghiên cứu

Nghiên cứu so sánh từ vựng của tin nhắn **spam** và tin nhắn thường (**ham**) bằng **TF-IDF**, rồi kiểm tra độ ổn định của kết quả khi đổi cách tiền xử lý văn bản. TF-IDF ở đây là công cụ **mô tả**: nó không phân loại spam/ham và không chứng minh quan hệ nhân quả.

| RQ | Câu hỏi (nguyên văn đã đăng ký) | Thao tác hóa |
|---|---|---|
| RQ1 | Which words and terms are most distinctive in spam messages compared to legitimate SMS messages? | Top-20 term mỗi lớp theo quy trình ở mục 5.3 (lọc DF rồi xếp hạng theo Δ). |
| RQ2 | How do the average TF-IDF weights of distinctive terms differ between spam and legitimate messages? | (a) Khác biệt Mean TF-IDF Δ là bao nhiêu? (b) Bao nhiêu của Δ đến từ prevalence (Δprev) và bao nhiêu từ intensity (Δint)? (c) Mức khác biệt về phân phối/rank (rank-biserial) là bao nhiêu? |
| RQ3 | How robust are the identified distinctive terms across different text preprocessing configurations? | Jaccard của Top-20 giữa 4 cấu hình, riêng từng lớp, trên 3 góc nhìn (mục 5.5). |

### Ánh xạ RQ ↔ hình/bảng

| RQ | Hình/bảng chính | Bổ trợ |
|---|---|---|
| Bối cảnh | Fig. 1, Table 1 | |
| RQ1 | Fig. 2, Table 2 | |
| RQ2 | Fig. 3a-b, Table 2, Table 3 | Fig. 3c (độ dài tin) |
| RQ3 | Fig. 4 | Bảng phụ lục (góc nhìn unigram chỉ chữ cái), Sensitivity S1 |

---

## 2. Dữ liệu

| Mục | Thông tin |
|---|---|
| Nguồn | UCI Machine Learning Repository, **SMS Spam Collection** (dataset 228): https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| Trích dẫn | T. A. Almeida and J. M. Gómez Hidalgo, "SMS Spam Collection," UCI Machine Learning Repository. |
| Quy mô gốc | 5.574 tin nhắn, nhãn `ham` / `spam`, file tab-separated, không có tiêu đề cột |
| Cách lấy | Tải zip từ trang UCI, giải nén, đặt file `SMSSpamCollection` vào `data/raw/`. **Không sửa file gốc.** |
| Giấy phép/chi phí | Miễn phí, tải công khai Creative Commons Attribution 4.0 (CC BY 4.0) theo trang UCI; ghi nguồn khi dùng lại |

### Xử lý tin trùng (tập chính = đã loại trùng)

- **Khóa trùng:** văn bản chuẩn hóa (lowercase, strip, gộp khoảng trắng). Giữ 1 bản đầu tiên của mỗi nhóm trùng.
- **Cùng khóa nhưng khác nhãn:** loại cả nhóm và báo cáo số lượng.
- **Không loại near-duplicate** (tin chỉ khác số điện thoại hoặc giải thưởng); ghi vào Limitations.
- **Tập chính:** corpus đã loại trùng (`data/processed/sms_dedup.csv`). **Sensitivity S1:** chạy lại cấu hình C2 trên corpus gốc (có trùng).
- Số liệu thực tế (điền sau khi chạy Bước 1):

| | Gốc | Sau loại trùng |
|---|---|---|
| Tổng số tin | 5.574 | 5.159 |
| Ham | 4.827 (86,6%) | 4.517 (87,6%) |
| Spam | 747 (13,4%) | 642 (12,4%) |
| Số nhóm trùng | 290 | |
| Số nhóm mâu thuẫn nhãn | 0 | |

---

## 3. Cấu trúc thư mục

```
sms-tfidf/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/SMSSpamCollection        # file gốc, không sửa
│   └── processed/sms_dedup.csv      # tập chính đã loại trùng
├── src/
│   ├── tfidf_lib.py                 # run_analysis và các hàm phụ
│   ├── figs.py                      # style + hàm vẽ Fig. 1-4
│   ├── conftest.py                  # fixture cho test
│   └── test_pipeline.py             # 5 test tự động
├── notebooks/main.ipynb             # notebook chính, chạy từ trên xuống dưới
├── figures/                         # PDF vector (và SVG nếu cần)
├── tables/                          # CSV xuất ra từ bảng tidy
├── paper/                           # main.tex, refs.bib
└── slides/                          # slides bảo vệ
```

---

## 4. Môi trường và cách tái lập

**Môi trường:** Python 3.12; các gói và phiên bản chính nằm trong `requirements.txt`
(pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, nltk, pytest, jupyterlab).

```bash
# 1. Tạo môi trường
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

# 2. Đặt dữ liệu: data/raw/SMSSpamCollection (xem mục 2)

# 3. Chạy test pipeline (5 test phải đều đạt)
cd src && pytest -q && cd ..

# 4. Chạy notebook từ đầu đến cuối
jupyter nbconvert --to notebook --execute notebooks/main.ipynb --output executed.ipynb
```

- **Seed:** 42 cho mọi bước ngẫu nhiên (mẫu đọc tin thật, split-sample nếu có).
- **Thứ tự tạo ra kết quả:** notebook đọc dữ liệu → loại trùng → `run_analysis` → xuất `tables/*.csv` → vẽ `figures/*.pdf`.
- **Lưu ý đường dẫn:** notebook dùng đường dẫn tương đối so với thư mục chứa notebook. Notebook tự tìm `data/raw/SMSSpamCollection` ở thư mục hiện tại hoặc thư mục cha, nên chạy được cả khi mở từ gốc repo lẫn từ `notebooks/`; đầu ra ghi vào `../tables/` và `../figures/`.

### Truy vết số liệu

Mọi hình và bảng sinh từ **bảng tidy** do hàm `run_analysis(df, config)` trả về, nên mọi con số trong paper truy được về một dòng số trong `tables/`.
Các cột chính: `term, side, idf, k_S, k_H, DF_S, DF_H, m_S, m_H, c_S, c_H, delta, d_prev, d_int, D, LPR`
và, khi bật suy diễn, `U, p_raw, p_holm, r, med_*, q1_*, q3_*, ci_lo_*, ci_hi_*`.

### Kiểm tra mã (5 test tự động trong `src/test_pipeline.py`)

| # | Kiểm tra | Mục đích |
|---|---|---|
| 1 | Toy corpus 3 tin: tính tay TF-IDF cho 1 term phổ biến và 1 term hiếm, so với scikit-learn | Pipeline đúng công thức |
| 2 | Mỗi hàng ma trận có chuẩn L2 = 1 | Xác nhận chuẩn hóa |
| 3 | `|m_g − p_g·c_g| < 1e-12` và `|Δ − (Δprev + Δint)| < 1e-12` với mọi term | Đồng nhất thức và phân rã |
| 4 | `analyzer("Win £1000 now!! txt me")` chứa `£`, `1000`, `!`, `txt` | Tokenizer giữ ký hiệu |
| 5 | `"call"` không nằm trong stopword NLTK (nhưng nằm trong danh sách của scikit-learn) | Tránh loại nhầm từ kỳ vọng của Spam |

Ngoài ra (thủ công): đọc 10 tin thật cho mỗi term trong một mẫu gồm term phổ biến và term hiếm; Restart kernel và Run All cuối ngày.

---

## 5. Phương pháp (đặc tả cố định, đăng ký trước khi chạy)

### 5.1 TF-IDF (scikit-learn `TfidfVectorizer`)

```
tf'(t,d) = 1 + ln tf(t,d)                    # sublinear_tf = True
idf(t)   = ln((1+n)/(1+df(t))) + 1           # smooth_idf = True, n = số tin trong corpus phân tích
x(t,d)   = tf'(t,d) * idf(t) / ||v_d||_2     # norm = "l2"
```

TfidfVectorizer được fit trên **toàn bộ** corpus phân tích; thống kê theo lớp tính sau đó.

| Tham số | Cấu hình chính (C2) |
|---|---|
| `lowercase` | True |
| `ngram_range` | (1, 2) |
| `min_df` | 5 (số tin tối thiểu trên toàn corpus) |
| `token_pattern` | `r"(?u)\b\w[\w']*\b\|[£$€!?]"` (giữ £, $, €, !, ?, số, "i'm") |
| `stop_words` | None (cấu hình chính); C3/C4 dùng danh sách **NLTK**, không dùng `stop_words="english"` của scikit-learn |
| `sublinear_tf`, `norm` | True, "l2" |
| seed | 42 |

### 5.2 Đại lượng theo lớp g ∈ {S (Spam), H (Ham)} và term t

| Ký hiệu | Định nghĩa |
|---|---|
| k_g(t), p_g(t) | k_g = số tin trong lớp g có x(t,d) > 0; p_g = DF_g(t) = k_g / n_g |
| m_g(t) | Mean TF-IDF = (1/n_g) Σ x(t,d) trên **mọi** tin của lớp (cả tin không chứa t) |
| c_g(t) | Conditional Mean = Σ x(t,d) / k_g, chỉ trên tin chứa t; quy ước c_g = 0 nếu k_g = 0 |
| Đồng nhất thức | m_g = p_g · c_g (đúng vì x > 0 khi và chỉ khi term có mặt); được `assert` trong code |
| Δ | Δ(t) = m_S(t) − m_H(t) |
| Phân rã | Δprev = (p_S − p_H)·(c_S + c_H)/2 ; Δint = (c_S − c_H)·(p_S + p_H)/2 ; Δ = Δprev + Δint **chính xác**, không có phần dư. Đây là phân rã số học, không phải nhân quả. |
| Cách báo cáo | Báo cáo giá trị tuyệt đối của hai thành phần; chỉ tính % đóng góp khi hai thành phần cùng dấu, nếu khác dấu ghi "-" |
| Đối chiếu không phụ thuộc IDF | D = p_S − p_H ; LPR = ln((p_S + ε)/(p_H + ε)), ε = 0,001 |

### 5.3 Chọn distinctive term (RQ1)

1. **Lọc:** với lớp g, giữ term có p_g(t) ≥ **2%** trong chính lớp đó (ghi số tin tuyệt đối tương ứng sau khi loại trùng).
2. **Xếp hạng:** phía Spam theo Δ giảm dần; phía Ham theo −Δ giảm dần. Lấy **N = 20** mỗi phía. Báo cáo DF của cả hai lớp.
3. DF là tiêu chí lọc, **không** phải bằng chứng thống kê. P-value **không** được dùng để quyết định term nào là distinctive.
4. Không so độ lớn Δ giữa phía Spam và phía Ham (IDF toàn corpus phụ thuộc tỷ lệ lớp). Hai phía vẽ ở hai panel riêng, trục riêng.

### 5.4 Suy diễn mô tả (RQ2)

- **Mann-Whitney U hai phía** cho từng term trong 40 term, trên vector x(t,d) của tin Spam vs Ham (gồm cả số 0); `scipy.stats.mannwhitneyu(..., alternative="two-sided", method="asymptotic")`. Diễn giải là khác biệt về phân phối/rank, không phải khác biệt trung bình.
- **Holm** trên một họ **40 phép thử** cố định (20 Spam + 20 Ham).
- **Effect size:** rank-biserial r = 2·U_S/(n_S·n_H) − 1 ∈ [−1, 1]; báo cáo thêm median và IQR kèm c_g.
- **Wilson 95% CI** cho p_S và p_H (`statsmodels.stats.proportion.proportion_confint(method="wilson")`), tính trên tập đã loại trùng.
- P-value và CI mang tính **khám phá**: term được chọn bằng chính dữ liệu, Holm chỉ sửa đa so sánh trong tập đã cố định, và near-duplicate còn sót. Không dùng cụm "có ý nghĩa thống kê" như bằng chứng xác nhận độc lập.
- Suy diễn chỉ chạy trên C2 (và S1).

### 5.5 Độ ổn định tiền xử lý (RQ3)

| Cấu hình | Dấu câu/ký hiệu | Stopwords | N-gram |
|---|---|---|---|
| C1 | Giữ (£ ! ? số) | Không loại | (1,1) |
| C2 (gốc) | Giữ | Không loại | (1,2) |
| C3 | Bỏ £ ! ? (chỉ chữ-số) | NLTK | (1,1) |
| C4 | Bỏ £ ! ? | NLTK | (1,2) |

- Mỗi cấu hình **fit lại độc lập** (IDF riêng) và chạy đúng quy trình 5.3. Tên gọi: **combined preprocessing sensitivity analysis**. Nghiên cứu không tuyên bố đã tách riêng tác động của stopwords hay dấu câu.
- **Jaccard** J(A,B) = |A∩B| / |A∪B|, tính riêng từng lớp, trên 3 góc nhìn: (i) toàn bộ term; (ii) chỉ unigram (lọc candidate về unigram trước khi xếp hạng); (iii) unigram chỉ gồm chữ cái. Góc nhìn (ii) là cơ sở diễn giải chính, vì unigram vs bigram làm J(i) thấp dù nội dung không đổi.
- **Sensitivity S1 (bắt buộc):** chạy lại C2 trên corpus gốc (có trùng): Jaccard Top-20, tỷ lệ cùng dấu của Δ, tương quan hạng của effect size.
- **Nếu còn thời gian:** S2 (ngưỡng DF 1% và 5%), one-factor-at-a-time (phụ lục), split-sample validation (một lần chia 50/50, stratified, seed 42, không chia lại).

---

## 6. Pre-registration

> Mục này được viết và commit **trước khi chạy bất kỳ phân tích TF-IDF nào**.
> **Commit đăng ký:** `600f245` · **Thời điểm:** `2026-11-26 18:06:35 +0700`

### 6.1 Giả thuyết

| ID | Giả thuyết | Điều kiện bác bỏ |
|---|---|---|
| H1 | Phía Spam chủ yếu gồm từ kêu gọi hành động/liên hệ/phần thưởng (call, free, txt, claim, prize, www, £, số). Phía Ham chủ yếu gồm từ hội thoại cá nhân (u, ok, i'm, ...). | Phần lớn Top-20 không thuộc các nhóm này theo codebook ở 6.2, với ngưỡng "phần lớn" ở 6.2 mục 4. |
| H2 | Với phần lớn (≥ 12/20) term phía Spam, \|Δprev\| > \|Δint\|, tức khác biệt chủ yếu do độ phổ biến. | Dưới 12/20 term thỏa. Δint có thể âm vì tin Spam dài hơn (chuẩn hóa L2); kết quả đó vẫn hợp lệ. |
| H3 | Jaccard Top-20 trên unigram ≥ 0,5 giữa mọi cặp cấu hình ở cả hai lớp; Jaccard trên toàn bộ term thấp hơn. | Ngược lại. Ngưỡng 0,5 là quy ước của nghiên cứu, không phải chuẩn học thuật. |

Giả thuyết bị bác bỏ vẫn là kết quả hợp lệ và sẽ được báo cáo như vậy.

### 6.2 Codebook gán nhãn cho H1

**Nguyên tắc**
- Đơn vị gán nhãn: mỗi term trong Top-20 của mỗi phía (40 term), **một nhãn duy nhất** cho mỗi term.
- Gán theo **chức năng của term trong tin nhắn**, không theo việc term xuất hiện ở phía nào.
- Khi mơ hồ: đọc 10 tin thật chứa term (`show_examples(clean, term, label, k=10)`, seed = 42) và chọn nhóm chiếm đa số. Vẫn không rõ: nhãn **E**; không ép vào nhóm khác.

**Các nhóm**

| Nhãn | Tên nhóm | Định nghĩa | Ví dụ minh họa (không phải dự đoán kết quả) |
|---|---|---|---|
| A | Kêu gọi hành động / liên hệ / thúc giục | Từ yêu cầu người nhận làm gì (gọi, nhắn, trả lời, nhận), kênh liên hệ, hoặc từ tạo áp lực thời gian | call, txt, text, reply, send, stop, claim, ring, contact, urgent, now |
| B | Phần thưởng / ưu đãi | Từ nói về quà, giải thưởng, tiền, khuyến mãi | free, prize, win, winner, cash, award, bonus, offer, voucher, gift, guaranteed |
| C | Ký hiệu / số / địa chỉ web | Term có **hình thức** là ký hiệu tiền tệ, dấu câu đặc biệt, chuỗi số, đầu số dịch vụ, tên miền | £, $, €, !, ?, www, http, com, uk, 1000, 150p |
| D | Hội thoại cá nhân | Từ đời thường giữa người quen: đại từ ngôi 1-2, viết tắt chat, xã giao, cảm xúc, sinh hoạt | u, ur, ok, i'm, me, my, lol, hi, thanks, sorry, love, home, later, tomorrow |
| E | Khác | Không khớp rõ nhóm nào hoặc đọc tin thật vẫn không xác định được | (tự do) |

**Quy tắc xử lý trường hợp khó**
1. **Bigram:** gán theo nhóm của thành phần mang nghĩa nội dung. Nếu hai thành phần thuộc hai nhóm khác nhau (ví dụ "free call": B và A), đọc tin thật và chọn nhóm chiếm đa số; nếu hòa, ưu tiên **C > B > A > D**.
2. **Nhóm C xét theo hình thức**, kể cả ở phía Ham ("?" ở phía Ham vẫn là C, không tính vào D).
3. **Số đứng riêng** (ví dụ "2", "4" có thể là viết tắt của "to/for"): đọc tin thật; viết tắt hội thoại thì D, số tiền/đầu số/giá thì C.
4. **Từ thúc giục** (urgent, now, today): xếp vào A (lựa chọn có chủ ý của nghiên cứu, không phải chuẩn học thuật).
5. **Từ nhiều nghĩa** (call, text, stop): theo nghĩa chiếm đa số trong 10 tin đã đọc, không theo nghĩa từ điển.
6. Tên riêng, địa danh, từ hiếm không rõ vai trò: E.

**Quy tắc đánh giá H1 (đã khóa)**
- Phía Spam: ủng hộ nếu **≥ 11/20** term thuộc A ∪ B ∪ C.
- Phía Ham: ủng hộ nếu **≥ 11/20** term thuộc D.
- H1 tổng thể chỉ được ủng hộ khi **cả hai phía** đạt ngưỡng; báo cáo từng phía riêng.
- Ngưỡng 11/20 (> 50%) là quy ước của nghiên cứu.

**Quy trình:** xuất `tables/codebook_labels.csv` (`side, term, category`) → gán nhãn theo codebook → commit "Codebook labels (before interpretation)" → đếm theo quy tắc trên.
(Tùy chọn) sau vài giờ, gán lại không nhìn nhãn cũ và báo cáo số term khác nhãn giữa hai lần.

**Nhật ký sai lệch so với codebook** (điền nếu có)

| Thời điểm | Sai lệch | Lý do | Ảnh hưởng đến H1 |
|---|---|---|---|
| 2026-11-27 | Nhãn codebook được điền và commit chung với bước đánh giá H1 (commit `1c3c23d`, "H1 evaluation"), không phải commit riêng "Codebook labels (before interpretation)" | Quy trình ban đầu yêu cầu commit riêng trước khi đếm | Không đổi codebook hay ngưỡng; kết quả H1 vẫn tính theo quy tắc đã khóa |

### 6.3 Tham số đã khóa

| Tham số | Giá trị |
|---|---|
| N mỗi lớp | 20 |
| Ngưỡng DF trong chính lớp | 2% |
| `min_df` (toàn corpus) | 5 |
| Họ Holm | 40 phép thử (20 Spam + 20 Ham) |
| ε của LPR | 0,001 |
| Seed | 42 |
| Tập chính | Corpus đã loại trùng; S1 = corpus gốc |
| Stopwords | Danh sách NLTK |

---

## 7. Kết quả (điền sau khi chạy)

Mọi con số dưới đây **phải khớp** các file trong `tables/`. Không điền từ trí nhớ.

| Mục | Vị trí | Trạng thái |
|---|---|---|
| Table 1: đặc điểm dataset, trùng lặp, độ dài | `tables/table1.csv` | Xong |
| Top-20 mỗi phía (bảng tidy C2) | `tables/top_terms_C2.csv` | Xong |
| Table 2: m, DF + Wilson CI, c, Δ, Δprev, Δint | `tables/table2_top_terms.csv` | Xong |
| Table 3: Mann-Whitney, p Holm, r, D, LPR | `tables/table3_inference.csv` | Xong (Holm < 0,05 ở 39/40 term) |
| Ma trận Jaccard | `tables/jaccard_*.csv` | Xong (3 góc nhìn × 2 lớp) |
| Sensitivity S1 | `notebooks/main.ipynb`, mục 5.2; tóm tắt ở `paper/findings.md` F10 | Xong (Jaccard 1,00 Spam / 0,90 Ham; cùng dấu Δ 100%) |
| Kết luận H1 / H2 / H3 | `findings.md` | H1 chỉ ủng hộ phía Spam (13/20), không ủng hộ phía Ham (10/20); H2 ủng hộ (18/20); H3 bị bác bỏ (Jaccard unigram min 0,48 Spam, 0,14 Ham) |

---

## 8. Ghi nhận việc sử dụng AI

> Phần này phải phản ánh **đúng những gì bạn thực sự đã làm**. Sửa mọi dòng cho khớp thực tế.

- **Công cụ AI đã dùng:** Claude (Anthropic)
- **Dùng cho việc gì:** gợi ý lộ trình 2 ngày, soạn code mẫu và hàm vẽ, soát lỗi và rà soát repo
- **Phần do tôi tự làm/quyết định:** chọn đề tài, chạy toàn bộ phân tích, gán nhãn codebook, đọc tin thật, diễn giải kết quả, viết paper
- **Cách tôi kiểm chứng đầu ra của AI:**
  - Chạy 5 test tự động; tính tay ví dụ mini 3 tin và so với scikit-learn.
  - Đối chiếu công thức trong code với mục 3 của Research Plan v2.
  - Đọc tin thật cho các term trong Top-20 để kiểm tra ngữ cảnh.
  - Đối chiếu từng con số trong paper/slides với `tables/`.
  - Tra lại mọi tài liệu tham khảo từ trang gốc, không chép từ AI. (xác nhận trước khi nộp)

---

## 9. Hạn chế (Limitations)

- TF-IDF là kỹ thuật mô tả: không phân loại spam/ham và không chứng minh quan hệ nhân quả; một từ có điểm cao chưa chắc là dấu hiệu chắc chắn của Spam.
- Kết quả phản ánh một dataset SMS tiếng Anh thu thập khoảng những năm 2000, chủ yếu từ Anh và Singapore; không khái quát cho SMS spam hiện nay.
- Term được chọn bằng chính dữ liệu dùng để kiểm định: P-value và CI mang tính khám phá, có thể lạc quan, và Holm không sửa được điều này. Near-duplicate chưa được loại.
- IDF tính trên corpus lệch lớp (Spam chiếm thiểu số) làm so sánh độ lớn Δ giữa hai phía bị bất đối xứng; chuẩn hóa L2 làm trọng số một term phụ thuộc thành phần tin nhắn.
- Lưới tiền xử lý là combined sensitivity analysis, không tách riêng tác động từng yếu tố (trừ phụ lục one-factor nếu có).
- Spam có ít quan sát hơn nên ước lượng cho Spam kém ổn định hơn Ham.

---

## 10. Thay đổi so với đề xuất ban đầu (Topic.pdf)

- RQ2 giữ nguyên câu chữ; bổ sung thao tác hóa prevalence/intensity.
- Biểu đồ: 6 → 4 hình chính + 1 panel; word cloud chuyển sang phụ lục; biểu đồ so sánh Spam-Ham tách thành hình Δ (có phân rã) và bảng.
- Tập chính là corpus đã loại trùng (Topic.pdf chỉ yêu cầu kiểm tra và báo cáo trùng).
- Chọn term theo Δ sau khi lọc DF (thay cho "TF-IDF trung bình cao nhất mỗi nhóm"); Fig. 2 đổi tên tương ứng.
- Bổ sung: đối chiếu DF/LPR, Wilson CI, rank-biserial, Jaccard riêng unigram.

---

## 11. Tài liệu tham khảo

1. T. A. Almeida and J. M. Gómez Hidalgo, "SMS Spam Collection," UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering: New collection and results," in *Proc. 11th ACM Symp. Document Eng.*, 2011, pp. 259-262.
3. G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Inf. Process. Manage.*, vol. 24, no. 5, pp. 513-523, 1988.
4. S. Robertson, "Understanding inverse document frequency: On theoretical arguments for IDF," *J. Documentation*, vol. 60, no. 5, pp. 503-520, 2004.
5. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, 2011.
6. S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*, vol. 6, no. 2, pp. 65-70, 1979.

Các nguồn đề xuất thêm trong plan (Mann-Whitney 1947; Wilson 1927; Newcombe 1998; Nothman et al. 2018) **cần tra lại thông tin từ trang gốc** trước khi đưa vào paper; chưa liệt kê ở đây cho đến khi được xác minh.

Phần mềm dùng trong phân tích (cần cite trong paper): scikit-learn, SciPy, statsmodels, NLTK, pandas, NumPy, Matplotlib. Phiên bản: scikit-learn 1.9.1, SciPy 1.18.1, statsmodels 0.15.0, NLTK 3.10.4, pandas 3.1.1, NumPy 2.5.5, Matplotlib 3.11.2.