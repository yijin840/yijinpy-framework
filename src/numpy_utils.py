import numpy as np


def numpy_arr():
    
    # 1. 创建一个零数组
    arr_zero = np.zeros((3, 3))
    print("Zero Array:", arr_zero)

    # 2. 创建一个单位矩阵
    arr_eye = np.eye(3)
    print("Identity Matrix:", arr_eye)

    # 3. 创建一个随机数组
    arr_random = np.random.rand(3, 3)
    print("Random Array:", arr_random)

    # 4. 创建一个指定范围的数组
    arr_range = np.arange(0, 10, 2)
    print("Range Array:", arr_range)

    # 5. 创建一个均匀分布的数组
    arr_linspace = np.linspace(0, 10, 5)
    print("Linspace Array:", arr_linspace)

    # 6. 数组加法
    arr_sum = arr_zero + arr_eye
    print("Array Sum:", arr_sum)

    # 7. 数组点积
    arr_dot = np.dot(arr_eye, arr_random)
    print("Dot Product:", arr_dot)

    # 8. 数组转置
    arr_transpose = arr_random.T
    print("Transpose Array:", arr_transpose)

    # 9. 计算数组的均值
    arr_mean = np.mean(arr_random)
    print("Mean of Array:", arr_mean)

    # 10. 计算数组的标准差
    arr_std = np.std(arr_random)
    print("Standard Deviation:", arr_std)

    # 11. 空数组
    arr_empty = np.empty((2, 2))
    print("Empty Array:", arr_empty)

    # 12. 常数数组
    arr_full = np.full((2, 2), 7)
    print("Full Array (value 7):", arr_full)


    # 13. 正态分布随机数
    arr_normal = np.random.randn(3, 3)
    print("Normal Distribution Array:", arr_normal)


    # 14. 指定值创建数组
    arr_const = np.array([1, 2, 3, 4])
    print("Custom Array:", arr_const)


    # 15. 获取数组形状
    print("Shape of arr_const:", arr_const.shape)


    # 16. 改变数组形状
    arr_reshaped = arr_const.reshape(2, 2)
    print("Reshaped Array:", arr_reshaped)


    # 17. 获取数组维度
    print("Dimensions of arr_const:", arr_const.ndim)


    # 18. 数组索引与切片
    arr_slice = arr_const[1:3]
    print("Array Slice:", arr_slice)


    # 19. 设置特定值
    arr_const[0] = 99
    print("Modified Array:", arr_const)


    # 20. 数组合并
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    arr_concat = np.concatenate([arr1, arr2])
    print("Concatenated Array:", arr_concat)


    # 21. 数组拆分
    arr_split = np.split(arr_concat, 2)
    print("Split Array:", arr_split)

    # 22. 元素按条件替换
    arr_conditioned = np.where(arr_const > 5, 10, arr_const)
    print("Conditioned Array:", arr_conditioned)

    # 23. 数组运算加法
    arr_const = np.array([1, 2, 3, 4]).reshape(2, 2)
    arr_full = np.full((2, 2), 7)
    arr_add = arr_const + arr_full
    print("Array Addition:", arr_add)

    # 24. 元素按位置乘法
    arr_multiply = arr_const * arr_full
    print("Element-wise Multiplication:", arr_multiply)


    # 25. 求数组的最大值和最小值
    arr_max = np.max(arr_full)
    arr_min = np.min(arr_full)
    print("Max Value:", arr_max, "Min Value:", arr_min)


    # 26. 数组的索引获取最大值位置
    arr_axis_max = np.max(arr_full, axis=0)
    print("Max Value by Axis:", arr_axis_max)

    # 27. 数组的索引获取最小值位置
    arr_axis_min = np.min(arr_full, axis=1)
    print("Min Value by Axis:", arr_axis_min)


    # 28. 按轴计算最大值
    arr_axis_max = np.max(arr_full, axis=0)
    print("Max Value by Axis:", arr_axis_max)


    # 29. 按轴计算最小值
    arr_min_index = np.argmin(arr_full)
    print("Min Value Index:", arr_min_index)


    # 30. 计算数组的累积和
    arr_cumsum = np.cumsum(arr_const)
    print("Cumulative Sum:", arr_cumsum)


    # 31. 计算数组的累积乘积
    arr_cumprod = np.cumprod(arr_const)
    print("Cumulative Product:", arr_cumprod)

    # 32. 求数组的方差
    arr_var = np.var(arr_full)
    print("Variance of Array:", arr_var)


    # 33. 求数组的中位数
    arr_median = np.median(arr_full)
    print("Median of Array:", arr_median)


    # 34. 随机打乱数组
    np.random.shuffle(arr_const)
    print("Shuffled Array:", arr_const)


    # 35. 取数组前n个最大值
    arr_top_n = np.partition(arr_full, -2)[-2:]
    print("Top 2 Values:", arr_top_n)


    # 36. 数组按照特定条件排序
    arr_sorted = np.sort(arr_const)
    print("Sorted Array:", arr_sorted)


    # 37. 生成一个对角矩阵
    arr_diag = np.diag([1, 2, 3])
    print("Diagonal Matrix:", arr_diag)


    # 38. 数组按条件筛选
    arr_filtered = arr_const[arr_const > 2]
    print("Filtered Array:", arr_filtered)

    # 39. 生成高斯分布的随机数
    arr_gaussian = np.random.normal(0, 1, size=(3, 3))
    print("Gaussian Array:", arr_gaussian)


    # 40. 使用矢量化计算两个数组的差异
    arr_diff = np.subtract(arr_full, arr_const)
    print("Array Difference:", arr_diff)


    # 41. 矩阵乘法
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    arr_matrix_mult = np.dot(arr1, arr2)
    print("Matrix Multiplication:", arr_matrix_mult)

    # 42. 元素平方
    arr_square = np.square(arr_const)
    print("Squared Array:", arr_square)


    # 43. 计算数组的求和
    arr_sum = np.sum(arr_const)
    print("Sum of Array:", arr_sum)


    # 44. 求数组的所有元素的积
    arr_prod = np.prod(arr_const)
    print("Product of Array:", arr_prod)


    # 45. 数组的标准化
    arr_normalized = (arr_const - np.mean(arr_const)) / np.std(arr_const)
    print("Normalized Array:", arr_normalized)


    # 46. 计算数组的指数
    arr_exp = np.exp(arr_const)
    print("Exponential Array:", arr_exp)


    # 47. 元素级别的三角函数
    arr_sin = np.sin(arr_const)
    print("Sine of Array:", arr_sin)


    # 48. 数组的逻辑运算
    arr_logical = arr_const > 2
    print("Logical Array:", arr_logical)


    # 49. 计算数组元素总数
    arr_size = np.size(arr_const)
    print("Array Size:", arr_size)


    # 50. 从数组中获取唯一值
    arr_unique = np.unique(arr_const)
    print("Unique Values in Array:", arr_unique)
