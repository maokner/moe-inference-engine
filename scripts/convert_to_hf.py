"""Convert miniMoE's custom checkpoint into a local Hugging Face directory."""

import argparse
import json

from moe_engine.hf.convert import convert_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/minimoe_sft.pt")
    parser.add_argument("--output", default="checkpoints/minimoe-hf")
    parser.add_argument(
        "--force", action="store_true", help="replace a non-empty output directory"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = convert_checkpoint(args.checkpoint, args.output, force=args.force)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
