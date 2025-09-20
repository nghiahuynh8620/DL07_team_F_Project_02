# DL07 Team F – Project 02

## 📖 Mô tả dự án

`DL07_team_F_Project_02` là một dự án học tập/do nhóm thực hiện, sử dụng Python notebook  xây dựng web app đưa ra dự đoán, khảo sát)*.

Một số điểm nổi bật:
- Sử dụng Jupyter Notebook (ví dụ: `Topic_2_Agoda_RS.ipynb`) để phân tích dữ liệu, xử lý và trực quan hóa.
- Có file `app.py` để triển khai ứng dụng web (GUI/API).
- File `java_bootstrap.py` hỗ trợ khởi tạo hoặc một module phụ trợ (nếu có).
- Các thư viện cần thiết được liệt kê trong `requirements.txt` và/hoặc `packages.txt`.
- Folder `data` chứa dữ liệu để phân tích; `output` / `outputs` chứa kết quả chạy và biểu đồ/báo cáo.

## 🚀 Cài đặt & chạy

1. **Tải về dự án**
   ```bash
   git clone https://github.com/nghiahuynh8620/DL07_team_F_Project_02.git
   cd DL07_team_F_Project_02
   ```

2. **Tạo môi trường ảo (virtual environment)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # trên Linux/macOS
   venv\Scripts\activate      # trên Windows
   ```

3. **Cài đặt các phụ thuộc**
   ```bash
   pip install -r requirements.txt
   ```

4. **Chạy ứng dụng Web**
   ```bash
   python app.py
   ```
   Sau đó mở trình duyệt tại `http://localhost:5000` (hoặc cổng được cấu hình).

5. **Chạy notebook phân tích**
   - Mở Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Chọn mở `Topic_2_Agoda_RS.ipynb` và thực thi các ô (cells) theo thứ tự.

## 🗂 Cấu trúc thư mục

```
DL07_team_F_Project_02/
├── .devcontainer/
├── data/
├── output/
├── outputs/
├── Topic_2_Agoda_RS.ipynb
├── app.py
├── java_bootstrap.py
├── logo.png
├── packages.txt
├── requirements.txt
└── README.md    ← bạn đang ở đây
```

- `data/`: dữ liệu đầu vào
- `output/` & `outputs/`: chứa kết quả đầu ra, biểu đồ, báo cáo
- `Topic_2_Agoda_RS.ipynb`: notebook phân tích dữ liệu
- `app.py`: ứng dụng Web/API nếu có giao diện
- `java_bootstrap.py`: module phụ trợ hoặc script hỗ trợ
- `logo.png`: logo hoặc hình ảnh liên quan dự án
- `requirements.txt` & `packages.txt`: các gói cần thiết cho môi trường

## 📊 Kết quả & minh họa

*(Chèn hình ảnh, screenshot nếu có, ví dụ giao diện web, biểu đồ từ notebook, v.v.)*

## 🛠 Công nghệ & thư viện sử dụng

- Python (phiên bản XYZ)  
- Flask (nếu dùng) để xây dựng ứng dụng Web  
- Pandas, NumPy, Matplotlib / Seaborn (hoặc các thư viện tương tự) cho phân tích dữ liệu và trực quan hóa  
- Jupyter Notebook  

## 👥 Đóng góp

Nếu bạn muốn đóng góp:

1. Fork repo này.
2. Tạo nhánh (branch) tính năng/chỉnh sửa của bạn: `feature/ten_tinh_nang`.
3. Commit & push lên fork.
4. Tạo pull request mô tả rõ thay đổi.

## ⚠️ Lưu ý

- Dữ liệu (`data/`) có thể lớn hoặc nhạy cảm, nên kiểm tra xem có cần ẩn thông tin cá nhân.
- Đảm bảo môi trường Python tương thích, tránh xung đột phiên bản thư viện.

## 📝 Liên hệ

- Tác giả hoặc nhóm: **Nghĩa Huỳnh** , **Nguyễn Ngọc Huy** et al.  
