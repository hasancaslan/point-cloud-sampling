for i in range(18):
    if i < 14:
        filename = f"./out/train-12_12_14-0{i}.txt"

    elif i < 16:
        filename = f"./out/test-12_12_14-0{i}.txt"

    else:
        filename = f"./out/val-12_12_14-0{i}.txt"

    f = open(filename, "r")
    lines = []
    for line in f.readlines():
        lines.append(line.strip())

    f = open(filename, "w+")
    for i in range(len(lines) - 1):
        for j in range(i + 1, len(lines)):
            f.write(lines[i] + " " + lines[j] + "\n")

    f.close()