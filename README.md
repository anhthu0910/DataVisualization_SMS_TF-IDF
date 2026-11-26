## 3. Methodology Specification (công thức và quy trình cố định)

### 3.1 TF-IDF (scikit-learn)

```python
tf*(t, d) = 1 + ln tf(t, d)          # sublinear_tf=True
idf(t) = ln((1 + n) / (1 + df(t))) + 1  # smooth_idf=True, n = number of docs in analysis corpus
x(t, d) = tf*(t, d) * idf(t) / ||v_d||_2  # norm="l2", v_d = tf-idf vector of doc d
```

Fit TfidfVectorizer on the WHOLE analysis corpus; class statistics computed afterwards.

| Tham số | Cấu hình chính (C2) | Ghi chú |
| :--- | :--- | :--- |
| lowercase / ngram_range | True / (1, 2) | Khớp Topic.pdf. |
| min_df | 5 | Số tin tối thiểu trên toàn corpus. Khác ngưỡng DF 2% dùng để chọn distinctive. |
| token_pattern (ví dụ) | r"(?u)\b\w[\w']*\b\|[£$!?]" | Giữ £, $, !, ?, số, "i'm". Phải kiểm tra bằng test (mục 4). |
| stop_words | None (cấu hình chính) | C3/C4 dùng danh sách NLTK. |
| random seed | 42 | Dùng cho mọi bước ngẫu nhiên (split, nếu có). |

### 3.2 Đại lượng theo lớp $g \in \{S, H\}$ và term $t$

| Ký hiệu | Định nghĩa |
| :--- | :--- |
| $k_g(t), p_g(t)$ | $k_g =$ số tin trong lớp $g$ có $x(t,d) > 0$; $p_g = DF_g(t) = k_g / n_g$. |
| $m_g(t)$ | Mean TF-IDF $= (1/n_g) \sum_{d \in g} x(t,d)$, tính cả tin không chứa $t$ ($x=0$). |
| $c_g(t)$ | Conditional Mean $= \sum_{d \in g} x(t,d) / k_g$, chỉ trên tin chứa $t$. Quy ước $c_g = 0$ nếu $k_g = 0$. |
| Đồng nhất thức | $m_g(t) = p_g(t) \cdot c_g(t)$. Đúng vì với sublinear_tf và smoothed idf, $x > 0$ khi và chỉ khi term có mặt. Assert trong code. |
| $\Delta_t$ | $m_S(t) - m_H(t)$. |
| Phân rã | $\Delta_{prev} = (p_S - p_H) \cdot (c_S + c_H)/2$; $\Delta_{int} = (c_S - c_H) \cdot (p_S + p_H)/2$. $\Delta_t = \Delta_{prev} + \Delta_{int}$ chính xác, không có số dư. |
| Cách báo cáo | Báo cáo giá trị tuyệt đối của hai thành phần. Chỉ tính % đóng góp khi hai thành phần cùng dấu; nếu khác dấu ghi "—" (tỷ lệ phần trăm khi mẫu số nhỏ hoặc dấu ngược nhau không ổn định). |
| Đối chiếu không phụ thuộc IDF | $D_t = p_S - p_H$; $LPR_t = \ln((p_S + \epsilon)/(p_H + \epsilon))$, $\epsilon = 0,001$ cố định. Hai đại lượng này chỉ dựa trên DF nên bất biến với IDF. |

### 3.3 Chọn distinctive term (RQ1)

* **Bước 1 – lọc:** với lớp $g$, giữ term có $p_g(t) \ge 2\%$ trong chính lớp đó. Số tin tối thiểu khác nhau giữa hai lớp (Spam ít tin hơn nhiều); ghi số tuyệt đối sau khi loại trùng.
* **Bước 2 – xếp hạng:** phía Spam theo $\Delta_t$ giảm dần; phía Ham theo $-\Delta_t$ giảm dần. Lấy Top-20 mỗi phía. Báo cáo DF của cả hai lớp.

DF là tiêu chí lọc, không phải bằng chứng thống kê. P-value không được dùng để quyết định term nào là distinctive.

Không so độ lớn $\Delta$ giữa phía Spam và phía Ham (IDF toàn corpus phụ thuộc tỷ lệ lớp; IDF luôn dương nên không đảo dấu $\Delta$ của một term, nhưng làm so sánh độ lớn giữa các term khác phía bị bất đối xứng). Vẽ hai phía ở hai panel riêng, trục riêng, có ghi chú.

### 3.4 Suy diễn mô tả (RQ2)

* Mann-Whitney U hai phía cho từng term trong 40 term, trên vector $x(t,d)$ của tin Spam vs Ham (gồm cả số 0): `scipy.stats.mannwhitneyu(..., alternative="two-sided", method="asymptotic")` (hiệu chỉnh ties). Diễn giải là khác biệt về phân phối/rank, không phải khác biệt trung bình; với dữ liệu nhiều số 0, kết quả chủ yếu phản ánh khác biệt prevalence.
* Holm trên một họ 40 phép thử, cố định trước.
* Effect size: rank-biserial $r = 2U_S / (n_S \cdot n_H) - 1 \in [-1, 1]$ ($r > 0$: giá trị Spam có xu hướng lớn hơn). Báo cáo thêm median và IQR (thường bằng 0; báo cáo kèm $c_g$).
* Wilson 95% CI cho $p_S$ và $p_H$ (`statsmodels.stats.proportion.proportion_confint(method="wilson")`), tính trên tập đã loại trùng. Tùy chọn: Newcombe CI cho $D_t$.

P-value và CI mang tính khám phá: term được chọn bằng chính dữ liệu, Holm chỉ sửa đa so sánh trong tập đã cố định, và near-duplicate còn sót. Không dùng cụm "có ý nghĩa thống kê" như bằng chứng xác nhận độc lập.

### 3.5 Độ ổn định tiền xử lý (RQ3)

| Cấu hình | Stopwords và dấu câu/ký hiệu | N-gram | Token pattern / stopwords |
| :--- | :--- | :--- | :--- |
| **C1** | Giữ | `(1, 1)` | Pattern chính, không loại stopword |
| **C2 (gốc)** | Giữ | `(1, 2)` | Pattern chính, không loại stopword |
| **C3** | Bỏ | `(1, 1)` | Chỉ chữ số (bỏ £ ! ?), stopwords NLTK |
| **C4** | Bỏ | `(1, 2)` | Chỉ chữ số (bỏ £ ! ?), stopwords NLTK |

Mỗi cấu hình fit lại độc lập (IDF riêng) và chạy đúng quy trình 3.3. Gọi tên chính xác: *combined preprocessing sensitivity analysis*; không tuyên bố đã tách riêng tác động của stopwords hay punctuation.

Jaccard $J(A, B) = |A \cap B| / |A \cup B|$, tính riêng từng lớp, trên 3 góc nhìn: (i) toàn bộ term; (ii) chỉ unigram (lọc candidate về unigram trước khi xếp hạng, để cả 4 cấu hình có đủ 20 unigram); (iii) unigram chỉ gồm chữ cái (tách hiệu ứng mất £, !, số). Unigram vs bigram khiến $J$(i) thấp mà nội dung không đổi, nên góc nhìn (ii) là cơ sở diễn giải chính.

* **Sensitivity S1 (bắt buộc):** chạy lại C2 trên corpus gốc (có trùng): Jaccard Top-20, tỷ lệ cùng dấu của $\Delta$, tương quan hạng của effect size.
* **S2 (nếu còn thời gian):** ngưỡng DF 1% và 5%, so Top-20.
* **Tùy chọn (phụ lục):** chạy *one-factor-at-a-time* từ C2 (đổi riêng stopwords, riêng punctuation, riêng n-gram) để quy nguyên nhân.

Thống kê suy diễn (3.4) chỉ chạy trên C2 (và S1). Jaccard chạy trên cả 4 cấu hình.

### 3.6 Split-sample validation (chỉ nếu còn thời gian)

Một lần chia 50/50, stratified theo nhãn, seed 42, ghi trước khi chạy; không chia lại. Tập đã loại trùng nên không có leakage giữa hai nửa.

* **Discovery:** fit vectorizer, tính DF/$\Delta$, lọc DF, chọn Top-20 mỗi phía.
* **Validation:** transform bằng vectorizer của Discovery (vocabulary + IDF giữ nguyên), kiểm định đúng 40 term, Holm, effect size.

Mỗi nửa chỉ còn khoảng một nửa số Spam nên power thấp. Không yêu cầu mọi term đạt P nhỏ; báo cáo effect size và tỷ lệ term cùng chiều $\Delta$.
