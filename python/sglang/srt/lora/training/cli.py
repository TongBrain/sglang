"""Internal CLI boundary; no platform UI or automatic cluster scheduling."""

import argparse
import json
import os

from .checkpoint import request_pause
from .jobs import JobStore, TrainingSpec
from .publish import InferenceClient, publish_job
from .runtime import execute_training, resume_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--spec", required=True)
    status = commands.add_parser("status")
    status.add_argument("--job-id", required=True)
    for command in ("pause", "resume"):
        control = commands.add_parser(command)
        control.add_argument("--job-id", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--job-id", required=True)
    publish.add_argument("--endpoint", required=True)
    publish.add_argument("--input-ids", type=json.loads, required=True)
    publish.add_argument("--canary-text", required=True)
    args = parser.parse_args()
    store = JobStore(args.job_root)
    if args.command == "train":
        with open(args.spec, encoding="utf-8") as stream:
            spec = TrainingSpec(**json.load(stream))
        result = execute_training(store, spec)
    elif args.command == "pause":
        result = request_pause(store, args.job_id)
    elif args.command == "resume":
        result = resume_training(store, args.job_id)
    elif args.command == "publish":
        result = publish_job(
            store,
            args.job_id,
            InferenceClient(
                args.endpoint,
                admin_key=os.environ.get("LORA_ADMIN_KEY"),
            ),
            args.input_ids,
            args.canary_text,
        )
    else:
        result = store.read(args.job_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
