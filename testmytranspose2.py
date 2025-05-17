import numpy as np
import pandas as pd
import torch

def mytranspose(x):   # 데이터 타입 : Matrix, Vector, DataFrame, Tensor
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