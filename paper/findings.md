# Findings (Bước 3.4) – cấu hình C2, tập đã loại trùng

**Nguồn số liệu:** `tables/top_terms_C2.csv`, `tables/table1.csv`, `tables/codebook_labels.csv` và output trong `notebooks/main.ipynb` (commit `1c3c23d`). F1–F8 đọc từ các file trên; F9–F11 lấy từ output mục 5 của notebook và được kiểm tra lại bằng cách chạy lại `run_analysis` trên `data/processed/sms_dedup.csv` và `data/raw/SMSSpamCollection` (5 test trong `src/test_pipeline.py` đều đạt).

**Cấu hình:** C2 = unigram + bigram, giữ £ $ € ! ?, không loại stopword; `min_df` = 5; lọc DF ≥ 2% trong chính lớp; N = 20 mỗi phía; Holm trên 40 phép thử; seed = 42.

**Cỡ mẫu (sau loại trùng):** n_Spam = 642, n_Ham = 4.517 (tổng 5.159). 2% của n_Spam = 12,84 → cần ≥ 13 tin Spam; 2% của n_Ham = 90,3 → cần ≥ 91 tin Ham.

---

## F1. Phía Spam: Top-20 và codebook (RQ1, H1 – phía Spam)

- **Describe:** Top-20 phía Spam theo Δ giảm dần bắt đầu bằng £, call, !, free, txt, to, your, stop, mobile, 2 và kết thúc bằng … prize, now, "now !". DF_S từ 9,5% ("now !") đến 62,3% ("to"); DF_H từ 0% (claim, prize) đến 25,5% ("to"). Codebook: A = 8, B = 2, C = 3, E = 7, D = 0. Số term thuộc A ∪ B ∪ C = **13/20** (ngưỡng đã khóa: ≥ 11).
- **Notice:** Có 5 term gần như chỉ xuất hiện ở Spam (DF_H ≤ 0,2%): claim (0%), prize (0%), www (0,02%), £ (0,11%), "now !" (0,15%). Mặt khác, 7 term nhãn E là từ chức năng hoặc chung (to, your, from, or, mobile, 1, 2) có DF_H từ 0,3% đến 25,5%. "now" và "now !" là n-gram chồng lấn, cùng nhãn A, nên không đếm như hai tín hiệu độc lập.
- **Limit:** Codebook do mình viết; gán nhãn có phần chủ quan (ví dụ "mobile", "1", "2" gán E). Nếu chuyển các term E này sang C thì tổng chỉ tăng, nên kết luận phía Spam không phụ thuộc vào các quyết định đó; nhưng bảng này chưa thể hiện tác động nếu codebook khác. Top-20 phụ thuộc cách xếp hạng theo Δ, nên chưa nói được đây là "top theo mọi tiêu chí".
- **Check next:** Chạy C1/C3/C4 (Bước 5) xem "to", "your", "from", "or" có còn trong Top-20 khi loại stopword NLTK (C3/C4) không; đọc 10 tin cho "mobile", "1", "2".
- **Kết luận H1 – phía Spam:** đạt ngưỡng (13 ≥ 11).

## F2. Phía Ham: Top-20 và codebook (RQ1, H1 – phía Ham)

- **Describe:** Top-20 phía Ham theo −Δ gồm i, my, me, ?, i'm, it, ok, in, but, that, so, lt, gt, not, "lt gt", how, come, you, at, ü. DF_H từ 3,0% (ü) đến 34,7% (i). Codebook: D = 10, E = 9, C = 1 ("?" theo quy tắc 2 của codebook). Số term thuộc D = **10/20** (ngưỡng đã khóa: ≥ 11).
- **Notice:** Phía Ham thiếu đúng 1 term so với ngưỡng. Ba term "lt", "gt", "lt gt" (DF_S = 0, DF_H = 4,96% / 4,96% / 4,36%) chiếm 3/20 vị trí và là cùng một chuỗi chồng lấn, nên thực chất là một tín hiệu; đã gán D. Chín term E (it, in, but, that, so, not, how, come, at) là từ chức năng thông thường.
- **Limit:** Kết quả sát ngưỡng (10 so với 11), nên nhạy với cách gán nhãn. Chưa đọc tin thật cho "lt"/"gt" trong notebook; cần xác nhận chúng là gì trước khi tin vào nhãn D (xem Check next). Không được đổi codebook hay ngưỡng sau khi đã thấy kết quả.
- **Check next:** Chạy `show_examples(clean, "lt", "ham", k=10)` và `show_examples(clean, "ü", "ham", k=10)`. Nếu nhãn D của "lt"/"gt" không phù hợp với 10 tin đọc được, ghi vào "Nhật ký sai lệch" ở README 6.2 thay vì sửa lặng lẽ; báo cáo cả hai phía riêng.
- **Kết luận H1 – phía Ham:** **không** đạt ngưỡng (10 < 11).

## F3. Kết luận H1 tổng thể

- **Describe:** Quy tắc đã khóa: H1 chỉ được ủng hộ khi cả hai phía đạt ngưỡng. Spam 13/20 (đạt), Ham 10/20 (không đạt).
- **Notice:** H1 **không được ủng hộ trọn vẹn**: phần về Spam đứng vững, phần về Ham thì không theo ngưỡng 11/20. Đây là kết quả hợp lệ và nên báo cáo đúng như vậy, kèm lưu ý Ham chỉ thiếu 1 term.
- **Limit:** Ngưỡng 11/20 là quy ước của nghiên cứu, không phải chuẩn học thuật; nhóm D còn bị chia sẻ với nhóm E cho các từ chức năng.
- **Check next:** Ghi lại cách báo cáo (từng phía riêng) trong paper; không điều chỉnh ngưỡng.

## F4. Phân rã Δ phía Spam (RQ2, H2)

- **Describe:** **18/20** term Spam có |Δ_prev| > |Δ_int| (ngưỡng đã khóa: ≥ 12). |Δ_prev| từ 0,0085 đến 0,0558; |Δ_int| từ 0,000007 đến 0,0132. Δ_int âm ở 18/20 term; chỉ "claim" (+0,0104) và "prize" (+0,0085) có Δ_int dương. Ở các term có Δ_int âm, Δ_prev lớn hơn chính Δ (trung vị của tỷ số Δ_prev/Δ trên 20 term là 1,13).
- **Notice:** Hai term có Δ_int dương là hai term có DF_H = 0. Với quy ước c_H = 0 khi không có tin Ham chứa term, thì Δ_int = (c_S − 0)·(p_S + 0)/2 > 0 một cách cơ học; vì vậy dấu dương này không nên đọc là "cường độ cao hơn ở Spam".
- **Limit:** Đây là phân rã số học, không phải nhân quả. Δ_int âm ở hầu hết term phù hợp với việc chuẩn hóa L2 làm giảm trọng số mỗi từ trong tin dài (trung vị độ dài Spam 148 so với Ham 53 ký tự), nhưng chưa có phép kiểm tra nào tách riêng ảnh hưởng này; cần giữ ở mức "phù hợp với", không viết là nguyên nhân.
- **Check next:** Vẽ Fig. 3a và 3c cạnh nhau; có thể so c_S với c_H trong các tin cùng độ dài. Báo cáo giá trị tuyệt đối của hai thành phần, chỉ tính % khi cùng dấu.
- **Kết luận H2:** **ủng hộ** (18 ≥ 12).

## F5. Phân rã Δ phía Ham (RQ2)

- **Describe:** **14/20** term Ham có |Δ_prev| > |Δ_int|. Δ_int âm ở 19/20 term, dương chỉ ở "but" (+0,0004). Δ của "you" = −0,0081 nhưng Δ_prev = +0,0049 (dương): DF_S 32,4% > DF_H 27,6%.
- **Notice:** "you" xếp trong Top-20 phía Ham theo −Δ chỉ vì c_H (0,125) lớn hơn c_S (0,082), tức vì thành phần cường độ, trong khi về độ phổ biến thì xuất hiện nhiều hơn ở Spam (D = +0,048, LPR = +0,159). Phía Ham ít thuộc về độ phổ biến hơn phía Spam (14/20 so với 18/20).
- **Limit:** Không so độ lớn Δ giữa hai phía (IDF toàn corpus phụ thuộc tỷ lệ lớp ≈ 12,4% Spam sau loại trùng). Các term "lt", "gt", "ü" có c_S = 0 theo quy ước (không có tin Spam chứa).
- **Check next:** Đưa "you" vào Discussion như ví dụ về Δ và DF nghịch dấu; đọc tin chứa "you" ở cả hai lớp.

## F6. Đối chiếu Δ với D và LPR (độc lập IDF)

- **Describe:** Trong 20 term mỗi phía, tương quan hạng Spearman giữa Δ và D là 0,76 (Spam) và 0,87 (Ham); giữa Δ và LPR xấp xỉ 0 (0,00 cho Spam, −0,10 cho Ham). Theo LPR, đầu bảng Spam là £ (5,05), claim (4,96), prize (4,70), www (4,65), txt (3,91); theo D, đầu bảng là call (0,39), to (0,37), ! (0,35), £ (0,33).
- **Notice:** Δ và D xếp hạng khá giống nhau (đều nghiêng về term phổ biến), còn LPR ưu tiên term gần như vắng ở lớp kia (claim, prize, www; phía Ham: gt, lt, "lt gt", ü). "to" xếp 20/20 theo LPR (0,89) nhưng 2/20 theo D.
- **Limit:** Chỉ có 40 term đã chọn theo Δ trong `top_terms_C2.csv`; chưa so được với Top-20 chọn trực tiếp theo D hoặc LPR trên toàn từ vựng. LPR với ε = 0,001 rất nhạy với term có DF thấp ở lớp đối diện.
- **Check next:** Dùng bảng đầy đủ `tab` (return_full=True) để lấy Top-20 theo D và theo LPR (sau cùng bộ lọc DF), rồi đếm giao với Top-20 theo Δ; đưa D, LPR vào Table 3.

## F7. Kiểm định, effect size và độ rộng CI (RQ2c)

- **Describe:** Holm < 0,05 ở **39/40** term (Holm lớn nhất trong 39 term là 0,0004). Ngoại lệ duy nhất là "you": p_Holm = 0,657, r = −0,009. Rank-biserial |r| từ 0,093 đến 0,383 ở phía Spam (lớn nhất: call, 0,383) và từ 0,009 đến 0,302 ở phía Ham (lớn nhất: i, 0,302). Wilson 95% CI của DF: trung bình rộng 0,061 cho DF_S của 20 term Spam so với 0,008 cho DF_H của chúng.
- **Notice:** Hiệu ứng nhìn chung nhỏ đến vừa dù P rất nhỏ; ví dụ "claim" có DF_S = 14,2% (91/642) và DF_H = 0 (0/4.517), nhưng r chỉ 0,142. "!" có 302/642 tin Spam và 530/4.517 tin Ham chứa nó.
- **Limit:** Term được chọn bằng chính dữ liệu kiểm định, nên P chỉ mang tính mô tả; Holm chỉ sửa đa so sánh trong tập 40 đã cố định chứ không sửa selection bias. Với dữ liệu nhiều số 0, Mann–Whitney chủ yếu phản ánh khác biệt prevalence. Near-duplicate chưa loại. CI của Spam rộng hơn vì n_Spam nhỏ (642).
- **Check next:** Báo cáo "< 0,001" thay vì số mũ cực nhỏ; không viết "có ý nghĩa thống kê" như bằng chứng xác nhận; split-sample nếu còn giờ.

## F8. Ngữ cảnh thật của term (kiểm tra 6)

- **Describe:** Đã đọc 10 tin Spam chứa "free" (và các tin chứa "call", trích đầu): "free" xuất hiện trong lời mời dịch vụ (FREEMSG, "1st Tone FREE", "free calling"), kèm đầu số, giá (150p), hoặc lệnh STOP/txt; một tin là quảng cáo không có lời kêu gọi gọi/nhắn ("The free music player will arrive shortly"). "call" ở Spam xuất hiện như động từ kêu gọi gọi một số cụ thể ("just call 08712103738 NOW!").
- **Notice:** Ngữ cảnh khớp với nhãn A/B ở trên. `show_examples` tìm theo chuỗi con, nên có tin khớp "FREEMSG" hoặc "5free" mà tokenizer có thể không tách thành token "free"; số tin đọc được không phải số tin tính vào DF.
- **Limit:** Mẫu 10 tin chỉ minh họa, không đại diện. Chưa đọc tin Ham chứa "call" (DF_H = 3,9%) nên chưa kiểm tra được nghĩa "gọi cho tôi" ở Ham (quy tắc 5 của codebook); chưa đọc tin có "£", "lt", "ü" trong output notebook.
- **Check next:** Chạy `show_examples(clean, "call", "ham")`, `"£"`, `"lt"`, `"ü"`; chọn 1–2 ví dụ ngắn cho Discussion (không dùng như bằng chứng định lượng).

---

## F9. Độ ổn định theo tiền xử lý: Jaccard Top-20 (RQ3, H3)

- **Describe:** Góc nhìn chính (ii) – chỉ unigram, J giữa các cấu hình (Fig. 4a–b; `tables/jaccard_unigram_*.csv`):
  - **Spam:** C1–C2 = 1,00; C3–C4 = 0,90; C1/C2 so với C3/C4 = 0,48 (13 term chung trên 27 term hợp: call, claim, free, mobile, prize, reply, stop, text, txt, www, 150p, 1, 2).
  - **Ham:** C1–C2 = 0,90; C3–C4 = 0,90; C1/C2 so với C3/C4 chỉ 0,14–0,18 (C1 và C3 chung 5 term: come, got, gt, lt, ok).
  - Góc nhìn (i) toàn bộ term (Fig. 4c–d): Spam C1/C2 so với C3/C4 = 0,43–0,48; Ham = 0,14–0,18; C1–C2 giảm còn 0,90 (Spam) và 0,82 (Ham) vì C2 có bigram. Góc nhìn (iii) chỉ chữ cái (bảng phụ lục): Spam C1–C2 = 0,82; Ham C1/C2 so với C3/C4 = 0,14.
- **Notice:** Hai lớp phản ứng rất khác nhau. Phía Spam giữ được lõi nội dung khi đổi cấu hình, nhưng mất £, !, to, your, from, or, now (C1 → C3) và thêm 6 term mới (4, cash, contact, service, uk, urgent). Phía Ham gần như bị thay hẳn: 15/20 term của C1 (i, my, me, ?, i'm, it, in, but, that, so, not, how, at, you, what) không còn trong C3, và C3 gồm các từ nội dung/chat (home, going, good, later, lor, da, k, ü…). Thêm bigram (C1 → C2, C3 → C4) hầu như không đổi tập unigram (Spam 1,00; Ham 0,90).
- **Limit:** Lưới 2×2 đổi đồng thời stopword NLTK và bỏ £ ! ?; chưa tách được từng yếu tố nên chỉ được nói "phù hợp với việc stopword loại các đại từ/từ chức năng ở phía Ham", không nói là nguyên nhân. Jaccard chỉ so tập Top-20, không so thứ hạng; Spam 0,48 sát ngưỡng 0,5 nên nhạy với đúng một vài term. Ngưỡng 0,5 là quy ước của nghiên cứu.
- **Check next:** (nếu còn giờ) one-factor-at-a-time từ C2 để tách stopwords, dấu câu, n-gram; đọc vài tin chứa term mới của C3 (urgent, cash, contact, uk). Trong paper, báo cáo hai lớp riêng và diễn giải theo góc nhìn (ii).

## F10. Sensitivity S1: corpus gốc (có trùng) so với đã loại trùng (RQ3)

- **Describe:** Chạy lại C2 trên corpus gốc (n_Spam = 747, n_Ham = 4.827): Jaccard Top-20 = **1,00** (Spam, 20/20 term chung) và **0,90** (Ham, 19/20 chung; tập đã loại trùng có "ü", tập gốc có "i'll"). Tỷ lệ cùng dấu Δ = **1,00** trên 40 term; Spearman của rank-biserial r giữa hai tập = **0,996** (39 term chung). Holm < 0,05 vẫn ở 39/40 term; "you" vẫn là ngoại lệ (p_Holm = 0,731, r = −0,006 so với 0,657 và −0,009).
- **Notice:** Việc loại 415 tin trùng (310 Ham, 105 Spam) hầu như không đổi danh sách term, hướng của Δ hay cỡ hiệu ứng.
- **Limit:** Chỉ loại trùng chính xác; near-duplicate còn nên S1 không cho biết kết quả có vững khi loại cả chúng. Tỷ lệ cùng dấu Δ tính trên 40 term đã chọn theo Δ nên dễ bằng 1. Không nên so P-value giữa hai tập (corpus gốc có prevalence và độ chính xác ảo).
- **Check next:** S2 (ngưỡng DF 1% và 5%) nếu còn giờ; nêu S1 trong Methods và Supplementary.

## F11. Kết luận H3

- **Describe:** Điều kiện đã khóa: Jaccard Top-20 unigram ≥ 0,5 giữa mọi cặp cấu hình ở cả hai lớp. Spam: min = 0,48 (< 0,5). Ham: min = 0,14 (< 0,5). Vế thứ hai (Jaccard toàn bộ term thấp hơn unigram): đúng ở một số cặp (Spam C1–C2 0,90 so với 1,00; Spam C2–C3 0,43 so với 0,48; Ham C1–C2 0,82 so với 0,90), bằng nhau ở các cặp còn lại, và không cặp nào cao hơn.
- **Notice:** H3 **bị bác bỏ** theo ngưỡng đã đăng ký, chủ yếu do phía Ham; phía Spam chỉ thiếu 0,02. Đây là kết quả hợp lệ và nên báo cáo từng lớp riêng.
- **Limit:** Không đổi ngưỡng sau khi đã thấy kết quả; ngưỡng 0,5 không phải chuẩn học thuật.
- **Check next:** Ghi H3 vào Results (chỉ mô tả) và Discussion (diễn giải ở mức "phù hợp với"); ghi vào "Nhật ký sai lệch" nếu có thay đổi quy ước.

---

## Việc cần ghi vào README / paper

- **Methods:** số tuyệt đối của ngưỡng DF 2% (≥ 13 tin Spam, ≥ 91 tin Ham); Top-20 (1,2) có n-gram chồng lấn ("now"/"now !", "lt"/"gt"/"lt gt"); cảnh báo stop_words/token_pattern ở C3/C4 nếu xuất hiện.
- **Results/Discussion:** H1 chỉ ủng hộ phía Spam (13/20), không ủng hộ phía Ham (10/20); H2 ủng hộ (18/20); "you" là ví dụ Δ và DF nghịch dấu, "you" nằm trong Top-20 phía Ham dù Δ_prev dương, và đây là ví dụ rất tốt cho Discussion; claim/prize có Δ_int dương do quy ước c_H = 0 chứ không phải cường độ cao hơn.
- **Results/Discussion (RQ3):** H3 bị bác bỏ (Spam min 0,48; Ham min 0,14); S1 gần như không đổi kết quả (Jaccard 1,00 / 0,90; cùng dấu Δ 100%).
- **Gate Ngày 1:** đã có kết quả cho cả 3 RQ (F1–F11), 5 test đạt, 4 hình nháp và Table 1–3.