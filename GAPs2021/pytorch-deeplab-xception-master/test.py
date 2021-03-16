import numpy as np
width = 4
height = 4
lbl = np.array([[0,2,3,1],[0,0,2,1],[2,1,0,0],[0,0,1,3]])
prob = np.array([0.5, 0.2, 0.2, 0.3]) #lbl_i表示的概率设置为prob_i
print(lbl)
print(prob)
print(prob[lbl])
print(prob[lbl].reshape(-1))
ps = np.exp(prob[lbl].reshape(-1))
ps /= np.sum(ps)
print(ps)
point = np.random.choice(width*height, 1, p=ps)
print(point)
w = point % width
h = point // width
print(h, w)