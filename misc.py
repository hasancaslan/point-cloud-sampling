from pyntcloud import PyntCloud

path = "./out"
filenames = []
for i in range(2):
    filenames.append(f"{path}/val-12_12_14-0{i}.txt")
    filenames.append(f"{path}/test-12_12_14-0{i}.txt")

for i in range(14):
    filenames.append(f"{path}/val-12_12_14-0{i}.txt")

for filename in filenames:
    f = open(filename, "r")

    for line in f.readlines():
        name = line.strip().split(".")[0]
        cloud = PyntCloud.from_file(f"{name}.ply")
        cloud.to_file(f"{name}.npz")