import sys
import json
from transformers import pipeline

def generate_math_solution(prompt, max_new_tokens=6000, temperature=1.0):
    """
    Uses a high-level transformers pipeline to generate a math problem solution using the GRPOtuned model.
    
    The GRPOtuned model is expected to output reasoning steps and a final answer in an XML format.
    
    Parameters:
      prompt (str): The input text prompt containing the math problem.
      max_new_tokens (int): Maximum number of tokens to generate. Default is 6000.
      temperature (float): Temperature for generation. Default is 1.2.
      
    Returns:
      str: The generated output.
    """
    # Create a text generation pipeline for the GRPOtuned model
    text_gen_pipe = pipeline("text-generation", model="HarleyCooper/GRPOtuned")
    # Generate output from the prompt
    output = text_gen_pipe(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    # Assuming the output is a list of dicts with the key 'generated_text'
    return output[0]['generated_text']

def save_result_to_jsonl(prompt, generated_output, filename="math_results.jsonl"):
    """
    Saves the prompt and the generated output to a JSONL file.
    
    Parameters:
      prompt (str): The math problem prompt.
      generated_output (str): The full generated output from the model.
      filename (str): The file path to save the JSONL entries. Default is "math_results.jsonl".
    """
    result = {
        "prompt": prompt,
        "generated_output": generated_output
    }
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Enter a math problem: ")
    solution = generate_math_solution(prompt)
    print("Generated Output:")
    print(solution)
    save_result_to_jsonl(prompt, solution)
    print("Result saved to math_results.jsonl")
