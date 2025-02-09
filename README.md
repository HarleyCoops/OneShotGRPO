# GRPOtuned Pipeline

A Python script for generating math solutions using the HarleyCooper/GRPOtuned model deployed on Hugging Face.

## Features

- Direct interface to HarleyCooper/GRPOtuned model
- Command-line and interactive input support
- Automatic result logging to JSONL format
- Configurable generation parameters

## Usage

The script can be used in two ways:

1. Command-line input:
```bash
python hf_grpotuned_pipeline.py "Your math problem here"
```

2. Interactive mode:
```bash
python hf_grpotuned_pipeline.py
# Then enter your math problem when prompted
```

## Function Details

### generate_math_solution(prompt, max_new_tokens=6000, temperature=1.0)
- Takes a math problem as input
- Uses Hugging Face pipeline for text generation
- Parameters:
  - prompt: The math problem text
  - max_new_tokens: Maximum length of generated response (default: 6000)
  - temperature: Controls randomness in generation (default: 1.0)

### save_result_to_jsonl(prompt, generated_output, filename="math_results.jsonl")
- Saves both input and output to a JSONL file
- Creates/appends to math_results.jsonl by default
- Stores each interaction as a JSON object

## Output

The script will:
1. Generate a solution using the model
2. Display the solution in the console
3. Save the interaction to math_results.jsonl

## Requirements

- transformers
- torch
- accelerate
- safetensors

## Note

The model will be downloaded from Hugging Face on first use and cached locally for subsequent runs.
