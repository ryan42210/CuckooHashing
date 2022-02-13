from os import spawnl
import matplotlib
import matplotlib.pyplot as plt


raw_lines= []
with open("plt3.txt") as f:
    raw_lines = f.readlines()


x_label = [i+"n" for i in raw_lines[0].split()]
mops2 = [float(i) for i in raw_lines[1].split()]
mops3 = [float(i) for i in raw_lines[2].split()]

w = 0.4

x = [2*(i+1) for i in range(len(x_label))]


plt.figure(figsize=(7,4))
plt.bar([i - w for i in x], mops2, label="2 hash function", color="black")
plt.bar([i + w for i in x], mops3, label="3 hash function", color="grey")

plt.xticks(x, x_label, size="small")
plt.xlabel("size of hash table")
plt.ylabel("Million operations per second (MOPS)")
plt.title("Performace on insertion with varying table size")
plt.ylim(bottom=700)
plt.legend()
plt.show()