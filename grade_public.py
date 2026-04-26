from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from judger import Judger


DEFAULT_RAW_RESPONSE_PATH = PROJECT_ROOT / "results" / "public_chunk_0.jsonl"
DEFAULT_RAW_RESPONSES_DIR = PROJECT_ROOT / "results"
DEFAULT_PUBLIC_DATA_PATH = PROJECT_ROOT / "data" / "public.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "public_results.jsonl"


def load_jsonl(path: Path) -> list[dict]:
	with path.open("r", encoding="utf-8") as handle:
		return [json.loads(line) for line in handle if line.strip()]


def extract_letter(text: str) -> str:
	match = re.search(r"\\boxed\{([A-Za-z])\}", text)
	if match:
		return match.group(1).upper()

	matches = re.findall(r"\b([A-Z])\b", text.upper())
	return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
	return extract_letter(response) == gold_letter.strip().upper()


def discover_chunk_paths(raw_responses_dir: Path) -> list[Path]:
	chunk_pattern = re.compile(r"^public_chunk_(\d+)\.jsonl$")
	chunk_items: list[tuple[int, Path]] = []

	for path in raw_responses_dir.iterdir():
		if not path.is_file():
			continue
		match = chunk_pattern.match(path.name)
		if not match:
			continue
		chunk_items.append((int(match.group(1)), path))

	chunk_items.sort(key=lambda item: item[0])
	return [path for _, path in chunk_items]


def grade_records(raw_response_path: Path, public_data_path: Path) -> list[dict]:
	public_items = {item["id"]: item for item in load_jsonl(public_data_path)}
	raw_items = load_jsonl(raw_response_path)

	judger = Judger(strict_extract=False)
	results: list[dict] = []

	for raw_item in raw_items:
		item_id = raw_item["id"]
		public_item = public_items.get(item_id)
		if public_item is None:
			raise KeyError(f"No public label found for id={item_id}")

		response = raw_item["response"]
		is_mcq = bool(public_item.get("options"))
		gold = public_item["answer"]

		if is_mcq:
			correct = score_mcq(response, str(gold))
		else:
			gold_list = gold if isinstance(gold, list) else [gold]
			try:
				correct = judger.auto_judge(
					pred=response,
					gold=gold_list,
					options=[[]] * len(gold_list),
				)
			except Exception:
				correct = False

		results.append(
			{
				"id": item_id,
				"is_mcq": is_mcq,
				"gold": gold,
				"response": response,
				"correct": correct,
			}
		)

	return results


def write_jsonl(records: list[dict], output_path: Path, mode: str = "w") -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open(mode, encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(results: list[dict]) -> None:
	mcq_results = [record for record in results if record["is_mcq"]]
	free_results = [record for record in results if not record["is_mcq"]]

	def accuracy(subset: list[dict]) -> float:
		return (sum(bool(record["correct"]) for record in subset) / len(subset) * 100) if subset else 0.0

	print("=" * 50)
	print("EVALUATION RESULTS")
	print("=" * 50)
	print(f"  MCQ        : {sum(bool(r['correct']) for r in mcq_results):4d} / {len(mcq_results):4d}  ({accuracy(mcq_results):.2f}%)")
	print(f"  Free-form  : {sum(bool(r['correct']) for r in free_results):4d} / {len(free_results):4d}  ({accuracy(free_results):.2f}%)")
	print(f"  Overall    : {sum(bool(r['correct']) for r in results):4d} / {len(results):4d}  ({accuracy(results):.2f}%)")
	print("=" * 50)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Grade public chunk responses using the starter notebook logic.")
	parser.add_argument("--raw-response-path", type=Path, default=DEFAULT_RAW_RESPONSE_PATH)
	parser.add_argument("--raw-responses-dir", type=Path, default=DEFAULT_RAW_RESPONSES_DIR)
	parser.add_argument("--public-data-path", type=Path, default=DEFAULT_PUBLIC_DATA_PATH)
	parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument(
		"--single-chunk",
		action="store_true",
		help="Grade only --raw-response-path instead of discovering all public_chunk_x.jsonl files.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.single_chunk:
		chunk_paths = [args.raw_response_path]
	else:
		chunk_paths = discover_chunk_paths(args.raw_responses_dir)
		if not chunk_paths:
			raise FileNotFoundError(
				f"No files matching public_chunk_x.jsonl found in {args.raw_responses_dir}"
			)

	all_results: list[dict] = []
	for idx, chunk_path in enumerate(chunk_paths):
		results = grade_records(chunk_path, args.public_data_path)
		write_jsonl(results, args.output_path, mode="w" if idx == 0 else "a")
		all_results.extend(results)
		print(
			f"Processed chunk {idx + 1}/{len(chunk_paths)}: {chunk_path.name} -> {len(results)} records"
		)

	summarize(all_results)
	print(f"Saved {len(all_results)} records to {args.output_path}")


if __name__ == "__main__":
	main()