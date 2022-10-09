from statistics import linear_regression
import pandas as pd
import numpy as np
from sklearn import linear_model, metrics, model_selection


# lấy dữ liệu từ tất cả các sheet trong file Folds5x2_pp.xlsx và chuyển sang numpy^^
data = pd.concat(
    pd.read_excel(
        "./Folds5x2_pp.xlsx",
        sheet_name=None,
    )
).to_numpy()

       
# tach du lieu train 70% va du lieu test 30%^^
data_train, data_test = model_selection.train_test_split(
    data,
    test_size=0.3,
)

x_test = data_test[:,:-1]
y_test = data_test[:,-1:]
    

# Sử dụng thuật toán KFOLD chia train data thành 3 phần, trong đó 2 phần để train và 1 phần validation^^
kfold = model_selection.KFold(
    n_splits=3,
)


# Biến min lưu lại giá trị sai số nhỏ nhất^^
MIN = 9999999999999999999999

for (train, valid) in kfold.split(data_train):
    x_train = data_train[train][:,:-1]
    y_train = data_train[train][:,-1:]
    
    x_val = data_train[valid][:,:-1]
    y_val = data_train[valid][:,-1:]
    
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(x_train, y_train)
    
    y_train_predicted = linear_regression.predict(x_train)
    y_val_predicted = linear_regression.predict(x_val)
    
    # Tổng hàm mất mát giữa train và validation^^
    sum_error = (
        metrics.mean_squared_error(y_train, y_train_predicted) + 
        metrics.mean_squared_error(y_val, y_val_predicted)
    )
    
    
    if sum_error < MIN:
        MIN = sum_error
        reg = linear_regression    

# W: Hệ số hồi quy^^
# W0: hệ số bias^^
print("W = ", reg.coef_)
print("W[0] = ", reg.intercept_)

data_predicted = reg.predict(x_test)

# print(data_predicted)
# Đánh giá độ chính xác^^
print(reg.score(x_test, y_test))