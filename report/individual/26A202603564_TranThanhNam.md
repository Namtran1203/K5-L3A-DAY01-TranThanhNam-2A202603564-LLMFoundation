# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

* **MSSV:** 2A202603564
* **Họ và tên:** Trần Thành Nam
* **SEED:** 309

## 1\. Bằng chứng

|Tệp|Có?|
|-|-|
|[`day01\_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output|☑|
|[`lab1\_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra|☑|
|[`hand\_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước|☑|

## 2\. Kết quả chính

|Phương pháp|F1 test|F1 SUB 40|p95 (ms/tin)|USD / 1k tin|
|-|:-:|:-:|:-:|:-:|
|1. BoW + LogReg (tự code)|0.896|0.900|0.09|$0.0|
|2. BoW + MLP (deep learning)|0.921|0.915|0.23|$0.0|
|3a. Encoder (MiniLM) + kNN|0.922|0.930|63.74|$0.0|
|3b. Encoder (MiniLM) + LogReg|0.960|0.955|41.62|$0.0|
|3c. Encoder (MiniLM) + MLP head|0.950|0.945|43.01|$0.0|

## 3\. Dự đoán và thực tế

1. **Loss curve của MLP (`dl\_curve`):**

   * *Đoán:* Cả train loss và val loss sẽ giảm đẹp về gần $0$.
   * *Thực tế:* Val loss bị chững lại rồi nhích tăng nhẹ từ epoch 7–8.
   * *Lý do:* Model MLP nhiều tham số quá nên bị học thuộc lòng (overfitting) trên tập train.
2. **Hệ số học tập quá lớn (`lr = 50`):**

   * *Đoán:* Loss sẽ nhảy nhót lung tung quanh một giá trị cao.
   * *Thực tế:* Loss bùng nổ thành `NaN` (tràn số) ngay mấy epoch đầu.
   * *Lý do:* Bước nhảy quá lớn làm chệch hẳn khỏi vùng hội tụ, hỏng luôn hàm mất mát.
3. **Tìm tin rác bằng ngữ nghĩa (`dl\_embed`):**

   * *Đoán:* Tin spam không có từ khóa nhạy cảm như "free", "win" sẽ bị đoán lầm thành tin thường.
   * *Thực tế:* Model vẫn bắt đúng là spam nhờ điểm tương đồng cosine cao.
   * *Lý do:* Quên mất là MiniLM bắt được ngữ nghĩa cả câu chứ không chỉ đếm từ rời rạc.

## 4\. Phân tích 50 case khó

* **Model tụt nhất:** `BoW + LogReg` ở nhóm tin spam ngụy trang (không dùng từ "free", "win" mà viết kiểu văn xuôi hoặc giả danh thông báo ngân hàng).
* **2 case ví dụ:**

  1. *Ham bị đoán nhầm thành Spam:* `"Please protect yourself from e-threats. SIB never asks for sensitive information like Passwords..."` — Model BoW thấy các từ như `"Passwords"`, `"threats"` là auto chấm điểm spam, không hiểu đây là tin nhắn cảnh báo an toàn.
  2. *Spam bị đoán nhầm thành Ham:* Tin nhắn quảng cáo dùng ký tự cố tình viết chèn khoảng trắng hoặc từ mới ngoài từ điển. Vectơ BoW trả về toàn số 0 nên model đành đoán đại theo lớp chiếm đa số (ham).

## 5\. Thử thách cá nhân

* **Đề theo SEED (309):** Tối ưu threshold (ngưỡng phân loại) trên tập val thay vì dùng mốc mặc định $0.5$.
* **Cách làm:** Chạy vòng lặp thử threshold từ $0.1$ đến $0.9$ (bước $0.05$) trên tập val để tìm điểm có F1 cao nhất.
* **Kết quả:** Ngưỡng tối ưu giảm xuống $0.35$, giúp F1 của LogReg tăng từ $0.896$ lên $0.915$.
* **Giải thích:** Do dữ liệu lệch (spam ít), hạ threshold giúp model "mạnh dạn" bắt các tin nghi vấn hơn, tăng mạnh recall mà không làm hại precision quá nhiều.

## 6\. Phương án cho sản phẩm thật

* **Chọn:** **3b. Encoder (MiniLM) + LogReg**.
* **Lý do:** Đáp ứng chuẩn độ trễ $p95 = 41.62\\text{ ms} < 50\\text{ ms}$, chi phí $0$ (chạy local) và F1 tốt nhất ($0.960$).
* **Đánh đổi:** Chấp nhận F1 thấp hơn một chút so với dùng các API LLM khủng, đồng thời phải tự chuẩn bị hạ tầng CPU/GPU nhẹ để chạy mô hình embedding.

## 7\. AI log và điều đã học

* **AI log:** Mình dùng AI để giải thích lại công thức backprop và xin gợi ý cú pháp PyTorch/NumPy khi bị lỗi shape. Sau đó mình tự viết lại code, chạy `\_run\_check` và tự chỉnh tham số cho qua bài test.
* **Điều học được:** Bài này giúp mình thấy rõ sự khác biệt giữa "đếm từ" (BoW) và "hiểu nghĩa" (Embedding). Việc đóng băng một encoder tốt rồi gắn thêm classifier nhẹ phía sau vừa nhanh vừa đem lại hiệu quả vượt trội so với tự train model từ đầu.

