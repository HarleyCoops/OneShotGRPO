from __future__ import annotations

import argparse
from pathlib import Path

from oneshot_grpo.inference.pipeline import (
    DEFAULT_MODEL_ID,
    build_text_generator,
    generate_math_solution,
    save_prompt_and_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the GRPO-tuned math model.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Math prompt to solve. If omitted, the prompt is read from stdin.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Model identifier to load from Hugging Face Hub.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6000,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="math_results.jsonl",
        help="Path to the JSONL file where prompts and answers will be appended.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = " ".join(args.prompt) if args.prompt else input("Enter a math problem: ")
    generator = build_text_generator(model_id=args.model)
    solution = generate_math_solution(
        prompt,
        generator=generator,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("Generated Output:")
    print(solution)
    save_prompt_and_response(
        prompt,
        solution,
        path=Path(args.output_jsonl),
        extra={"model": args.model},
    )
    print(f"Result saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
