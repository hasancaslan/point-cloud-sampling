for i in range(5):
    filename = f"./out/train-12_12_14-0{i}.txt"
    f = open(filename, "r")

    lines = []
    for line in f.readlines():
        lines.append(line.strip())

    f = open(filename, "w+")
    for i in range(len(lines) - 1):
        for j in range(i + 1, len(lines)):
            f.write(lines[i] + " " + lines[j] + "\n")

    f.close()