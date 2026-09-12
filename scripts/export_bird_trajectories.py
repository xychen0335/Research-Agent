"""Convert local rollout JSONL to the trajectory format expected by BIRD-RL."""

import argparse
import json
from pathlib import Path


def convert(records, expected_records=None):
    trajectories = []
    seen = set()
    for position, record in enumerate(records):
        instance_idx = record.get("instance_idx")
        if not isinstance(instance_idx, int) or instance_idx < 0:
            raise ValueError("record %d has no valid instance_idx" % position)
        if instance_idx in seen:
            raise ValueError("duplicate instance_idx %d" % instance_idx)
        seen.add(instance_idx)
        prediction = record.get("prediction")
        trajectory = []
        if isinstance(prediction, str) and prediction.strip():
            call = {
                "name": "submit_solution",
                "arguments": {"sql": prediction.strip()},
            }
            trajectory.append({
                "action": "<tool_call>%s</tool_call>" % json.dumps(call, ensure_ascii=False),
                "end_flag": True,
            })
        trajectories.append({"instance_idx": instance_idx, "trajectory": trajectory})
    if expected_records is not None:
        expected = set(range(expected_records))
        if seen != expected:
            missing = sorted(expected - seen)
            unexpected = sorted(seen - expected)
            raise ValueError(
                "instance_idx coverage mismatch: missing=%r unexpected=%r"
                % (missing[:20], unexpected[:20])
            )
    return trajectories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL produced by python -m sql_agent.run")
    parser.add_argument("--output", required=True, help="BIRD-RL trajectory JSONL")
    parser.add_argument("--expected-records", type=int)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trajectories = convert(records, args.expected_records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in trajectories),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(trajectories), "output": str(output)}))


if __name__ == "__main__":
    main()
