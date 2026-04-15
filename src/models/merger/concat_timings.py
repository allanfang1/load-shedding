from pathlib import Path
import csv


def merge_all_csv_in_dir(input_dir: Path, output_file: Path) -> None:
    csv_files = sorted(
        file_path
        for file_path in input_dir.glob("*.csv")
        if file_path.resolve() != output_file.resolve()
    )

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    with output_file.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        wrote_header = False

        for csv_file in csv_files:
            with csv_file.open("r", newline="", encoding="utf-8") as current_file:
                reader = csv.reader(current_file)
                header = next(reader, None)

                if header is None:
                    continue

                if not wrote_header:
                    writer.writerow(header)
                    wrote_header = True

                writer.writerows(reader)


if __name__ == "__main__":
    input_dir = Path.cwd()
    output = input_dir / "timings_merged.csv"

    merge_all_csv_in_dir(input_dir, output)
    print(f"Merged CSV saved to: {output}")
