import argparse


def main():
    parser = argparse.ArgumentParser(description="Oxford Data XYZ Column Reorganizer")
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Input XYZ File",
        metavar="inp.xyz"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Output XYZ File",
        metavar="out.xyz"
    )
    args = parser.parse_args()

    with open(args.input_file, 'r') as inp, open(args.output_file, 'w') as outp:
        for line in inp:
            _, _, x, y, z, _ = filter(lambda l: l != '', map(lambda l: l.strip(), line.split()))
            outp.write(f"{x} {y} {z}\n")


if __name__ == '__main__':
    main()
