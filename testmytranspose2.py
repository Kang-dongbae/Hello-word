import numpy as np
import pandas as pd
import torch

def mytranspose(x):   # 데이터 타입 : Matrix, Vector, DataFrame, Tensor
    if isinstance(x, np.ndarray): # Matrix, Vector
        # 빈 행렬
        if x.size == 0:
            return np.empty((0, 0))
        # 1-dimension, [1, 2]
        if x.ndim == 1:
            return x.reshape(-1, 1)
        # 나머지
        y = np.ones((x.shape[1], x.shape[0])) # 동일환 크기의 행렬 생성
        for i in range(x.shape[0]): # 행 반복
            for j in range(x.shape[1]): # 열 반복
                y[j, i] = x[i, j] # 값 할당
        return y

# 예제 데이터 생성
x = np.array([[1, 2, 3], [4, 5, 6]])
transposed_x = mytranspose(x)

# 결과 출력
print(transposed_x)

# (1) Matrix의 경우
myvar1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])  # 5x2 행렬
print(mytranspose(myvar1))

myvar1 = np.empty((0, 0))  # 빈 행렬
print(mytranspose(myvar1))

myvar1 = np.array([[1, 2]])  # 1x2 행렬
print(mytranspose(myvar1))

myvar1 = np.array([[1], [2]])  # 2x1 행렬
print(mytranspose(myvar1))

# (2) Vector의 경우
myvar2 = np.array([1, 2, np.nan, 3])  # NA는 numpy의 NaN으로 표현
print(mytranspose(myvar2))

myvar2 = np.array([np.nan])  # 단일 NaN 값
print(mytranspose(myvar2))

myvar2 = np.array([])  # 빈 배열
print(mytranspose(myvar2))