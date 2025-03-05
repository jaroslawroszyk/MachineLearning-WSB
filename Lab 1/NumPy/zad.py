import numpy as np

# zad 1
# asarray
# lista = [1, 2, 3]
# arr = np.asarray(lista)
# print(type(arr))

# ones_like
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# ones_arr = np.ones_like(arr)
# print(ones_arr)

# zeros_like
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# lista = [1, 2, 3]

# zeros_arr = np.zeros_like(lista)
# print(zeros_arr)


# full

# lista = [1, 2, 3]
# full_arr = np.full((2, 3), 7)

# print(full_arr)

# full_like
# lista = [1, 2, 3]
# full_arr = np.full_like((2, 3), 7)

# print(full_arr)

#eye
# eye_arr = np.eye(3)
# print(eye_arr)

#identity
# identity_arr = np.identity(3)
# print(identity_arr)



# zad 2
# x = np.array([1, 2, 2.5, 0])
# print(x.astype(float))

#zad 3

# x = np.array([1.2, 2.7, 2.5, 2.1])
# print(f"x: {x}")


# x_str = x.astype(str)
# print(f"x_str: {x_str}")

# x_as_orginal = x_str.astype(float)
# print(f"x_as_orginal: {x_as_orginal}")


# zad 4
# arr_3d = np.ones((2, 3, 4))

# arr_2d = np.array([[10, 20, 30, 40],
#                    [50, 60, 70, 80],
#                    [90, 100, 110, 120]])

# result = arr_3d + arr_2d

# print("Tablica po operacji:\n", result)


# zad 5

arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

#sum
# print(arr_2d.sum())
# print(arr_2d.sum(axis=0))


#mean
# print(arr_2d.mean())

#std
# print(arr_2d.std())

#var
# print(arr_2d.var())

#min 
# print(arr_2d.min())

#max
# print(arr_2d.max())

#argmin
# print(arr_2d.argmin())

#argmax
# print(arr_2d.argmax())

# cumsum
# print(arr_2d.cumsum())

#cumprod
# print(arr_2d.cumprod())

# zad 8
# vec = np.array([1,2,3])

# vec_repeated = np.repeat(vec, repeats=4)

# result = np.tile(vec_repeated, reps=3)
# print(result)


# zad 9
# vec = np.arange(start=1, stop=10.25, step=0.25)
# print(vec)

# zad 10
# vec = np.linspace(start=1, stop=10, num=43)
# print(vec)

# zad 11
# vec = np.linspace(start=1, stop=10, num=43)
# print(vec[::3])

# zad 12
# Each element of this array is subtracted by 1
# tab = np.arange(6) - 1
# print(tab)

# zad 13

# vector1 = np.array([1, 2, 3, 4, 5, 6])
# matrix1 = vector1.reshape(3, 2)

# print("wypełnienie jej wierszowo wektorem \n", matrix1)

# matrix2 = vector1.reshape(2, 3).T
# print("wypełnienie jej kolumnowo wektorem \n", matrix2)

# vector2 = np.array([1, 2, 3])
# vector3 = np.array([4, 5, 6])

# matrix3 = np.vstack((vector2, vector3)).T

# print("zlożenie wektorów (1,2,3) i (4,5,6), \n", matrix3)

# vector4 = np.array([1, 4])
# vector5 = np.array([2, 5])
# vector6 = np.array([3, 6])

# matrix4 = np.column_stack((vector4, vector5, vector6)).T

# print("złożenie wektorów (1,4), (2,5) i (3,6). \n", matrix4)

# zad 14

# A = np.arange(1, 37).reshape(6, 6)

# B1 = A[::2, ::2][:, ::-1]

# print("Macierz B1:\n", B1)

# B2 = A[4:, :]

# print("\nMacierz B2:\n", B2)

# B3 = A[:, 2:4]

# print("\nMacierz B3:\n", B3)


# zad 15
# T = np.array([[5,2,1,6],[4,8,3,1],[2,4,7,8]])
# T.sort(axis=1)
# print(T)

# zad 16
# T = np.array([[5,2,1,6],[4,8,3,1],[2,4,7,8]])
# T.sort(axis=0)
# print(T)

# zad 17
# tab = np.array([[5,2,1,6],[4,8,3,1],[2,4,7,8]])

# sorted_indices = np.argsort(tab, axis=1)
# print(sorted_indices)


# zad 18
# T = np.array([[5,2,5,2,7,4],[2,2,1,5,4,3],[1,3,4,1,4,5],[1,3,1,1,2,2]])

# sorted_T = T[np.lexsort((T[:, 5], T[:, 3], T[:, 1]))]

# print(sorted_T)

# zad 19
# T = np.array([[5,2,5,2,7,4],[2,2,1,5,4,3],[1,3,4,1,4,5],[1,3,1,1,2,2]])

# split_T = np.hsplit(T, indices_or_sections=3)

# print(split_T[0])

# for i, part in enumerate(split_T):
#     print(f"Część {i+1}:\n{part}\n")

# zad 20

# T = np.array([[5,2,5,2,7,4],[2,2,1,5,4,3],[1,3,4,1,4,5],[1,3,1,1,2,2]])
# split_T = np.hsplit(T, indices_or_sections=[1,4,5])

# for i, part in enumerate(split_T):
#     print(f"Część {i+1}:\n{part}\n")

# zad 21
# T = np.array([[5,2,5,2,7,4],[2,2,1,5,4,3],[1,3,4,1,4,5],[1,3,1,1,2,2]])

# vsplit_T = np.vsplit(T, indices_or_sections=[2,3])
# print(vsplit_T)

# zad 22

# T = np.array([[5, 2, 5, 2, 7, 4], [2, 2, 1, 5, 4, 3], [1, 3, 4, 1, 4, 5], [1, 3, 1, 1, 2, 2]])

# split_T_hsplit = np.hsplit(T, 3)

# T_combined_hsplit = np.hstack(split_T_hsplit)

# split_T_hsplit_4 = np.hsplit(T, [1, 4, 5])

# T_combined_hsplit_4 = np.hstack(split_T_hsplit_4)

# split_T_vsplit = np.vsplit(T, [2, 3])

# T_combined_vsplit = np.vstack(split_T_vsplit)

# print("Po scaleniu po hsplit (3 części):")
# print(T_combined_hsplit)

# print("\nPo scaleniu po hsplit z 4 częściami:")
# print(T_combined_hsplit_4)

# print("\nPo scaleniu po vsplit (3 części):")
# print(T_combined_vsplit)

#zad 23
# w = np.arange(1, 11)

# w = w[:, np.newaxis]
# print(w)

# zad 24
# w = np.arange(1, 11)
# w = w[np.newaxis, :]
# print(w)


# zad 25
# w = np.arange(1, 11)

# multiplication_table = w * w[:, np.newaxis]

# print(multiplication_table)

# zad 26

# T = np.linspace(start=1, stop=10, num=43)

# condition = (T >= 2) & (T < 7)

# indices = np.nonzero(condition)

# print(indices[0])


# zad 27
# w = np.array([1, 2, 2, 2, 3, 1, 3, 1, 4, 4, 5, 6, 5, 7, 8, 8])

# unique_elements, indices, counts = np.unique(w, return_index=True, return_counts=True)

# print("Unique elements:", unique_elements)
# print("Indices of first occurrences:", indices)
# print("Number of repetitions:", counts)

# zad 28
# T = np.array([[1,2],[2,1],[2,2],[2,2],[2,3],[3,2],[1,2],[1,4],[2,1]])

# unique_T, indices, counts = np.unique(T, axis=0, return_index=True, return_counts=True)

# print("Unique elements:", unique_T)
# print("Indices of first occurrences:", indices)
# print("Number of repetitions:", counts)
