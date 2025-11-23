#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from openvino.runtime import serialize
from openvino.tools.mo import convert_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ONNX models in `models/` to OpenVINO IR files."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing the original ONNX models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ov-models"),
        help="Directory where converted IR files will be stored",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Target precision for converted models",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of ONNX filenames to convert. Defaults to all *.onnx",
    )
    return parser.parse_args()


def convert_all(models, output_dir, compress_to_fp16=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for model_path in models:
        if not model_path.is_file():
            continue
        print(f"[INFO] Converting {model_path} -> {output_dir}")
        ov_model = convert_model(
            input_model=str(model_path), compress_to_fp16=compress_to_fp16
        )
        dst = output_dir / model_path.stem
        serialize(ov_model, str(dst.with_suffix(".xml")), str(dst.with_suffix(".bin")))
        converted.append(dst)
    return converted


def main():
    args = parse_args()
    if not args.models_dir.exists():
        print(f"[ERROR] Models directory {args.models_dir} does not exist", file=sys.stderr)
        return 1
    if args.models:
        model_paths = [args.models_dir / m for m in args.models]
    else:
        model_paths = sorted(args.models_dir.glob("*.onnx"))
    if not model_paths:
        print("[WARN] No ONNX models found to convert", file=sys.stderr)
        return 1
    converted = convert_all(
        model_paths, args.output_dir, compress_to_fp16=args.precision == "fp16"
    )
    print(f"[INFO] Converted {len(converted)} model(s) into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
