# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def load_data(path="./data.txt"):
    lines = [i.split(" ") for i in open(path).read().strip().split("\n")]
    return lines


data = load_data()
x = [i for i in range(1, int(data[0][0]) + 1)]
y1, y2, y3 = [], [], []
for i in data[1:int(data[0][0]) + 1]:
    y1.append(float(i[0]))
    y2.append(float(i[1]))
    y3.append(float(i[2]))

plt.figure(figsize=(18, 8))

ax1 = plt.subplot(1, 2, 1)
ax1.yaxis.set_ticks_position('right')
plt.grid()
plt.plot(x, y1)
plt.xlabel("# of epoch")
plt.ylabel("loss")
plt.title("loss")
plt.ylim(0)
plt.yticks(np.arange(0, 5, 0.2))

ax2 = plt.subplot(1, 2, 2)
ax2.yaxis.set_ticks_position('right')
plt.grid()
plt.plot(x, y2, label='train')
plt.plot(x, y3, label="test")
plt.xlabel("# of epoch")
plt.ylabel("accuracy")
plt.title("accuracy")
plt.ylim(0)
plt.yticks(np.arange(0, 101, 5))

plt.legend()
plt.savefig("./analysis.jpg")
plt.show()
