# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def load_data(path="./data.txt"):
    lines = [i.split(" ") for i in open(path).read().strip().split("\n")]
    return lines


data = load_data()
x = range(1, int(data[0][0]) + 1)
y1, y2, y3 = [], [], []
for i in data[1:]:
    y1.append(i[0])
    y2.append(i[1])
    y3.append(i[2])

plt.figure(figsize=(18, 8))
plt.figure(1)
y_major_locator = MultipleLocator(len(x)//15)

ax1 = plt.subplot(121)
plt.plot(x, y1)
plt.xlabel("# of epoch")
plt.ylabel("loss")
plt.title("loss")
ax1.yaxis.set_major_locator(y_major_locator)

ax2 = plt.subplot(122)
plt.plot(x, y2, label='train')
plt.plot(x, y3, label="test")
plt.xlabel("# of epoch")
plt.ylabel("accuracy")
plt.title("accuracy")
ax2.yaxis.set_major_locator(y_major_locator)

plt.legend()
plt.savefig("./analysis.jpg")
plt.show()
