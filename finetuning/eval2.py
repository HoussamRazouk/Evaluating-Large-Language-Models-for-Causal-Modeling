import os
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
import gc

os.chdir("/home/feline/Evaluating-Large-Language-Models-for-Causal-Modeling/finetuning")

def load_model_and_tokenizer(checkpoint_path, load_in_8bit=True, load_in_4bit=False):
    """
    Loads the model and tokenizer from a checkpoint.
    You can adjust quantization parameters (8-bit or 4-bit) as needed.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    quantization_kwargs = {}
    if load_in_8bit:
        quantization_kwargs['load_in_8bit'] = True
    elif load_in_4bit:
        quantization_kwargs['load_in_4bit'] = True
        quantization_kwargs['bnb_4bit_quant_type'] = 'nf4'  # or 'fp4'
        quantization_kwargs['bnb_4bit_use_double_quant'] = True

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map='auto',
        **quantization_kwargs
    )

    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map='auto',
        return_full_text=False  # so that only new tokens are returned
    )
    return tokenizer, model, generator

def construct_chat_prompt(row):
    """
    Constructs the chat prompt using the exact text from your non-finetuned tests.
    This prompt will be used for every evaluation sample, regardless of domain.
    """
    domain = row["domain"]
    first_text = row["Text1"]
    second_text = row["Text2"]

    # System message
    system_message = (
        f"You are an expert in causality and {domain}. Your task is to help users model their domain knowledge by identifying if two texts describe the same causal variable. "
        "Texts that describe different values or the same value of a causal variable should be indicated."
    )

    # User message – note the exact wording as in your non-finetuned tests.
    user_message = (
        "Your task is to assess if the following two texts belong to the same causal variable. \n"
        "- If the two texts belong to the same causal variable, provide the variable name.\n"
        "- If the two texts are similar but do not belong to the same variable set the variable name to '', provide your explanation.\n"
        "Structure your answer as a JSON object including string 'Text1', string 'Text2', boolean 'Predicted Same Causal Variable', "
        "string 'Predicted Variable Name', and string 'Explanation'.\n\n"
        f"First text: ```{first_text}```\n"
        f"Second text: ```{second_text}```"
    )
    
    # Combine messages in a chat-like format (using special tokens) so the prompt is identical to the non-finetuned tests.
    prompt = f"<|system|> {system_message}\n<|user|> {user_message}\n<|assistant|> "
    return prompt

def main():
    # === Configuration ===
    # Path to your trained model checkpoint
    CHECKPOINT_PATH = "lora_70b_en_v1"  # UPDATE with your checkpoint path
    # Input CSV file with evaluation examples. This file should have at least the columns:
    # Text1, Text2, Same Causal Variable, Variable Name, model Name, domain
    INPUT_CSV = "sampled_data_set_large.csv"  # UPDATE with your evaluation CSV path
    # Output CSV file for saving results
    OUTPUT_CSV = "evaluation_results_v1.csv"
    # Batch size (adjust according to your GPU memory)
    BATCH_SIZE = 1
    # Maximum number of new tokens to generate
    MAX_NEW_TOKENS = 200

    # Load model, tokenizer, and generator pipeline
    tokenizer, model, generator = load_model_and_tokenizer(
        CHECKPOINT_PATH, load_in_8bit=False, load_in_4bit=True
    )
    prediction_model_name = os.path.basename(CHECKPOINT_PATH)

    # Load the evaluation CSV (all domains will be processed)
    df = pd.read_csv(INPUT_CSV)

    output_rows = []
    num_samples = len(df)
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Processing {num_samples} samples in {num_batches} batches...")

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        batch_df = df.iloc[start_idx:end_idx]

        # Create a prompt for each row using the same non-finetuned format
        prompts = [construct_chat_prompt(row) for _, row in batch_df.iterrows()]

        try:
            responses = generator(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,        # Use deterministic generation for fair comparison
                temperature=0.0,
                num_return_sequences=1
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            responses = ["" for _ in prompts]

        for i, response in enumerate(responses):
            # The response may be returned as a list/dict or a string, so handle accordingly.
            if isinstance(response, list) and response:
                generated_text = response[0].get('generated_text', '')
            elif isinstance(response, dict):
                generated_text = response.get('generated_text', '')
            elif isinstance(response, str):
                generated_text = response
            else:
                generated_text = ""

            # Remove the prompt part if it is present in the output.
            prompt_text = prompts[i]
            answer_text = generated_text[len(prompt_text):].strip() if generated_text.startswith(prompt_text) else generated_text.strip()

            # Try to parse the answer as a JSON object.
            try:
                parsed_answer = json.loads(answer_text)
                predicted_same_causal = parsed_answer.get("Predicted Same Causal Variable", "")
                predicted_variable_name = parsed_answer.get("Predicted Variable Name", "")
                explanation = parsed_answer.get("Explanation", "")
            except Exception as e:
                print(f"Failed to parse JSON for sample index {start_idx + i}: {e}")
                predicted_same_causal = ""
                predicted_variable_name = ""
                explanation = answer_text  # Save the raw output in case JSON parsing fails

            # Save results for each sample with the required columns.
            row_input = batch_df.iloc[i]
            output_rows.append({
                "Text1": row_input["Text1"],
                "Text2": row_input["Text2"],
                "Generated Same Causal Variable": row_input["Same Causal Variable"],
                "Predicted Same Causal Variable": predicted_same_causal,
                "Generated Variable Name": row_input["Variable Name"],
                "Predicted Variable Name": predicted_variable_name,
                "Data Generation Model": row_input["model Name"],
                "Prediction Model": prediction_model_name,
                "Domain": row_input["domain"],
                "Explanation": explanation
            })

    # Save the aggregated results to CSV in the requested format.
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_CSV, index=False, sep=",", encoding="utf-8")
    print(f"Evaluation results saved to {OUTPUT_CSV}")

    # Cleanup: unload the model to free up GPU memory.
    del model, tokenizer, generator
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
