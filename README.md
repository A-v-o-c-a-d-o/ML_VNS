# ML_VNS
Environment:
  - numpy==1.23.5
  - pandas==1.5.2
  - Pillow==9.2.0
  - requests==2.28.1
  - scikit-learn==1.2.0
  - scipy==1.9.3
  - torch==1.13.0
  - torchvision==0.14.0
  - transformers==4.25.1
  - underthesea==1.3.5
  - python==3.10.8
  
Các thư viện:
Pillow, requests: load và biểu diễn ảnh
transformers: load model PhoBert
underthesea: tokenize đầu vào với word segmentation
pandas: load data csv

Khi chạy chỉ cần chạy file scripts.py

Nếu chưa có model (model.pt, cnn.pt) ở trong package Model thì trong file scripts.py phải khởi tạo model mới:
Thay đổi:
  model = torch.load('Model/model.pt').to(device)
  cnn = torch.load('Model/cnn.pt').to(device)
bằng:
  model = MyModel().to(device)
  cnn = Basic_CNN_Module().to(device)

Khi train sẽ in ra màn hình accuracy của 2 quy trình cả train lẫn test (data ở đây sẽ lấy ở full_train.csv)
Scripts file sẽ tự động train và run mô hình dự đoán rồi lưu file submission.csv vào package Results (data ở đây sẽ lấy ở test.csv)

Ngoài ra ở đây còn có file vns.ipynb cũng có quy trình tương tự
