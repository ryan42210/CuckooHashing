raw_lines= []
with open("in.txt") as f:
    raw_lines = f.readlines()

result = [[],[],[]]

for line in raw_lines:
  data = line.split()
  result[0].append(data[0])
  result[1].append(data[1])
  result[2].append(data[2])


for label in result[0]:
  print(label, end=" ")
print()

# for time in result[1]:
#   print(time, end=" ")
# print()

for mops in result[2]:
  print(mops, end=" ")
print()