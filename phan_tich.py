import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. ĐỌC DỮ LIỆU ---
df = pd.read_csv('financial_anomaly_data.csv')

# Xóa bỏ các dòng bị trống dữ liệu để tránh lỗi 'float'
df = df.dropna(subset=['TransactionType']) 

# --- 2. XỬ LÝ DỮ LIỆU ---
target = 'TransactionType'

# Vẽ biểu đồ phân phối loại giao dịch
plt.figure(figsize=(10, 6))
sns.countplot(x=target, data=df)
plt.title('Phân phối các loại giao dịch tài chính')
plt.xticks(rotation=45)
plt.savefig('bieu_do_moi.png')
print("Đã lưu biểu đồ vào file bieu_do_moi.png")

# Chuyển đổi nhãn chữ sang số cho cột mục tiêu
le = LabelEncoder()
y = le.fit_transform(df[target])

# Chỉ giữ lại các cột số cho biến độc lập X
X = df.select_dtypes(include=['number'])
X = X.drop(columns=['TransactionID'], errors='ignore')

# --- 3. XÂY DỰNG MÔ HÌNH ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# --- 4. ĐÁNH GIÁ ---
y_pred = model.predict(X_test)
print(f"\n--- BÁO CÁO KẾT QUẢ DỰ ĐOÁN {target.upper()} ---")
print(f"Độ chính xác tổng thể (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")

# In báo cáo chi tiết mà không cần vẽ thêm biểu đồ gây lỗi
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred))