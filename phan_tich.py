import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("--- 1. ĐỌC VÀ HIỂU DỮ LIỆU ---")
# Đọc dữ liệu từ file Excel (header=1 vì dòng đầu tiên trong file của UCI là chú thích)
# Lưu ý: Đảm bảo file excel nằm cùng thư mục với file code này
df = pd.read_excel('default of credit card clients.xls', header=1)

# Xóa cột ID vì không có ý nghĩa trong việc dự đoán
df = df.drop('ID', axis=1)

# Đổi tên cột mục tiêu cho ngắn gọn dễ thao tác
df = df.rename(columns={'default payment next month': 'default'})

print(f"Số lượng mẫu dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
print("\nThông tin các cột:\n", df.info())

print("\n--- 2. PHÂN TÍCH DỮ LIỆU (EDA) ---")
# Tỷ lệ nhãn mục tiêu (Có vỡ nợ hay không)
print("Tỷ lệ vỡ nợ (0 = Không, 1 = Có):\n", df['default'].value_counts(normalize=True) * 100)

# Vẽ biểu đồ phân phối nhãn (Lưu biểu đồ này lại để dán vào báo cáo)
plt.figure(figsize=(6, 4))
sns.countplot(x='default', data=df)
plt.title('Phân phối khách hàng vỡ nợ thẻ tín dụng')
plt.xlabel('Vỡ nợ (0 = Không, 1 = Có)')
plt.ylabel('Số lượng khách hàng')
plt.savefig('bieu_do_vo_no.png') # Code tự động lưu ảnh ra thư mục
print("Đã lưu biểu đồ thành file bieu_do_vo_no.png")

print("\n--- 3. XÂY DỰNG MÔ HÌNH DỰ ĐOÁN ---")
# Tách dữ liệu thành biến độc lập (X) và biến mục tiêu (y)
X = df.drop('default', axis=1)
y = df['default']

# Chia tập dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Đã huấn luyện xong mô hình Random Forest!")

print("\n--- 4. ĐÁNH GIÁ KẾT QUẢ ---")
y_pred = model.predict(X_test)

# In ra các chỉ số đánh giá để dán vào báo cáo
print(f"Độ chính xác tổng thể (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nBáo cáo chi tiết (Classification Report):")
print(classification_report(y_test, y_pred))
