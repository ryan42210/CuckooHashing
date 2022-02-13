result = [[],[],[]]

raw_lines = []
with open("../in.txt") as f:
    raw_lines = f.readlines()

for line in raw_lines:
    data = line.split()
    result[0].append(int(data[0]))
    if (data[1] != "insert"):
        result[1].append("{:.3f}".format(float(data[1])))
        result[2].append("{:.1f}".format(float(data[2])))
    else:
        result[1].append("fail")
        result[2].append("--")

print("\\textbf{max eviction num", end="} & ")
for i in result[0]:
    print(i, end=" & ")

print("\\\\")
print("\\midrule")
print("t/ms(2 hash func)", end=" & ")
for time in result[1]:
    print(time, end=" & ")

print("\\\\")
print("MPOS(2 hash func)", end=" & ")
for mops in result[2]:
    print(mops, end=" & ")
print("\\\\")