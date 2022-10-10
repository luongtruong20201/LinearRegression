from sklearn import (
    linear_model,
    metrics,
    model_selection
)

import pandas as pd
import numpy as np

# lấy dữ liệu từ tất cả các sheet trong file Folds5x2_pp.xlsx
data = pd.concat(
    pd.read_excel(
        './Folds5x2_pp.xlsx',
        sheet_name=None,
    )
).to_numpy()

data_train, data_test = model_selection.train_test_split(
    data,
    test_size=0.3,
    train_size=0.7,
)

# lấy x_test, y_test
x_test = data_test[:, :-1]
y_test = data_test[:, -1:]

# sử dụng KFold chia train_dât thành 3 phần, 2 phần dùng cho huấn luyện mô hình
# 1 phần dùng cho validation

kfold = model_selection.KFold(
    shuffle=False,
    random_state=None,
    n_splits=5
)

MIN = 1e1000000000000000000
reg = ''

for (train_index, val_index) in kfold.split(data_train):
    x_train = data_train[train_index][:, :-1]
    y_train = data_train[train_index][:, -1:]

    x_val = data_train[val_index][:, :-1]
    y_val = data_train[val_index][:, -1:]

    linear = linear_model.LinearRegression()
    linear.fit(
        x_train,
        y_train
    )

    y_train_predicted = linear.predict(x_train)
    y_val_predicted = linear.predict(x_val)

    # Tính tổng mất mát giữa train và validation
    sum_error = (
        metrics.mean_squared_error(y_train, y_train_predicted) +
        metrics.mean_squared_error(y_val, y_val_predicted)
    )

    if sum_error < MIN:
        MIN = sum_error
        reg = linear

# W: Hệ số hồi quy^^
# W0: hệ số bias^^
print("W = ", reg.coef_)
print("W[0] = ", reg.intercept_)

data_predicted = reg.predict(x_test)

# print(data_predicted)
# Đánh giá độ chính xác^^
print("^^: ", reg.score(x_test, y_test))