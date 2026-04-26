from pathlib import Path
import re


root = Path(__file__).resolve().parent
results_dir = root / "results"
output_path = results_dir / "private_results.jsonl"


def chunk_key(path: Path) -> int:
	match = re.search(r"private_chunk_(\d+)\.jsonl$", path.name)
	return int(match.group(1)) if match else 10**9


chunk_paths = sorted(results_dir.glob("private_chunk_*.jsonl"), key=chunk_key)

with output_path.open("w", encoding="utf-8") as output_file:
	for chunk_path in chunk_paths:
		with chunk_path.open("r", encoding="utf-8") as input_file:
			for line in input_file:
				if line.strip():
					output_file.write(line if line.endswith("\n") else line + "\n")

print(f"Wrote {output_path} from {len(chunk_paths)} chunk files")