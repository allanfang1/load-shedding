from pathlib import Path
import csv


def concat_csv_files(file1: Path, file2: Path, output_file: Path) -> None:
    with file1.open("r", newline="", encoding="utf-8") as f1, \
         file2.open("r", newline="", encoding="utf-8") as f2, \
         output_file.open("w", newline="", encoding="utf-8") as out:

        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(out)

        header1 = next(reader1, None)
        header2 = next(reader2, None)

        if header1 != header2:
            raise ValueError("CSV headers do not match.")

        if header1 is not None:
            writer.writerow(header1)

        writer.writerows(reader1)
        writer.writerows(reader2)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    file_a = base_dir / "timings2.csv"
    file_b = base_dir / "timings3.csv"
    output = base_dir / "timings_merged.csv"

    concat_csv_files(file_a, file_b, output)
    print(f"Merged CSV saved to: {output}")
