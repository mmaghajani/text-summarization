import numpy as np
from numpy import linalg as LA
# import pandas as pd
#
# a = np.array([1, 2, 3, 4, 5])
# l = list()
# b = np.array([1, 1, 1, -21, -1])
#
# for item in b:
#     l.append(item)
#     # print(item)
# print(l)
# print(a - b)  # is ok
# c = 0
#
# for i in range(0, 10):
#     c += 2 * a
#
# print(b)
# print(LA.norm(b, 1))
# d = {'a': [1,2,3], 'b': [3,4,7]}
#
# f = pd.DataFrame([k, *v] for k, v in d.items())
# print(f)
#
# g = pd.DataFrame.from_dict(d,orient = 'index').reset_index()
# print(g)

#
# a = "علی به خانه آمد"
# b = "نیس"
# if b in a:
#     print("is in")


import os
rootdir = "data/Multi"   #/Track1/Source/D91A01/"

for i in range(1,9):
    subDir = os.path.join(rootdir, "Track"+str(i)+"/Source/")
    for root, dirs, files in os.walk(subDir):
        if dirs:
            for dir in dirs:
                for file in os.listdir(subDir+dir):
                    print(file)
                print("--------")
