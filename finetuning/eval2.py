import os
import json
import re
import gc
import torch
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig, 
    StoppingCriteria, 
    StoppingCriteriaList
)
from datasets import load_dataset

# Change working directory if needed.
os.chdir("/home/feline/Evaluating-Large-Language-Models-for-Causal-Modeling/finetuning")

# --- Custom batched stopping criteria ---
class StopOnCompleteJSON(StoppingCriteria):
    """
    Stops generation once each sequence in the batch has a balanced JSON object.
    For each sequence, it looks for the first '{' and then counts opening and closing braces.
    Generation stops only when all sequences have a complete JSON object.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        finished = []
        for seq in input_ids:
            decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
            try:
                start = decoded.index("{")
                json_part = decoded[start:]
                count = 0
                complete = False
                for char in json_part:
                    if char == "{":
                        count += 1
                    elif char == "}":
                        count -= 1
                    if count == 0 and char == "}":
                        complete = True
                        break
                finished.append(complete)
            except ValueError:
                finished.append(False)
        # Only stop when all sequences in the batch are complete.
        return all(finished)

# --- Helper function to fix invalid JSON values ---
def fix_invalid_json(text):
    # Replace standalone NaN with null.
    text = re.sub(r'\bNaN\b', 'null', text)
    # Replace True/False (outside of quotes) with true/false.
    text = re.sub(r'(?<!")\bTrue\b(?!")', 'true', text)
    text = re.sub(r'(?<!")\bFalse\b(?!")', 'false', text)
    return text

# --- Load model and tokenizer ---
def load_model_and_tokenizer(checkpoint_path, load_in_8bit=True, load_in_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map='auto',
        quantization_config=bnb_config,
    )
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map='auto',
        return_full_text=False  # Only new tokens are returned.
    )
    return tokenizer, model, generator

# --- Construct chat prompt (exactly as before) ---
def construct_chat_prompt(row):
    # Use the "domain" field; it will be renamed later.
    domain = row["domain"] if "domain" in row else row["Domain"]
    first_text = row["Text1"]
    second_text = row["Text2"]
    system_message = (
        f"You are an expert in causality and {domain}. Your task is to help users model their domain knowledge by identifying if two texts describe the same causal variable. "
        "Texts that describe different values or the same value of a causal variable should be indicated."
    )
    user_message = (
        "Your task is to assess if the following two texts belong to the same causal variable. \n"
        "- If the two texts belong to the same causal variable, provide the variable name.\n"
        "- If the two texts are similar but do not belong to the same variable set the variable name to '', provide your explanation.\n"
        "Structure your answer as a JSON object including string 'Text1', string 'Text2', boolean 'Predicted Same Causal Variable', "
        "string 'Predicted Variable Name', and string 'Explanation'.\n\n"
        f"First text: ```{first_text}```\n"
        f"Second text: ```{second_text}```"
    )
    prompt = f"<|system|> {system_message}\n<|user|> {user_message}\n<|assistant|> "
    return prompt

# --- Process a batch of samples ---
def process_batch(batch):
    prompts = []
    # Construct prompts for each example in the batch.
    for i in range(len(batch["Text1"])):
        row = {
            "Text1": batch["Text1"][i],
            "Text2": batch["Text2"][i],
            "domain": batch["domain"][i] if "domain" in batch else batch["Domain"][i]
        }
        prompts.append(construct_chat_prompt(row))
    
    responses = generator(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        num_return_sequences=1,
        stopping_criteria=StoppingCriteriaList([StopOnCompleteJSON(tokenizer)])
    )
    
    predicted_same_causal_list = []
    predicted_variable_name_list = []
    explanation_list = []
    
    for i, response in enumerate(responses):
        if isinstance(response, list) and response:
            generated_text = response[0].get('generated_text', '')
        elif isinstance(response, dict):
            generated_text = response.get('generated_text', '')
        elif isinstance(response, str):
            generated_text = response
        else:
            generated_text = ""
        
        prompt = prompts[i]
        answer_text = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()
        answer_text = fix_invalid_json(answer_text)
        try:
            parsed_answer = json.loads(answer_text)
            predicted_same_causal = parsed_answer.get("Predicted Same Causal Variable", "")
            predicted_variable_name = parsed_answer.get("Predicted Variable Name", "")
            explanation = parsed_answer.get("Explanation", "")
        except Exception as e:
            print(f"Failed to parse JSON for sample with Text1: {batch['Text1'][i]} and Text2: {batch['Text2'][i]}: {e}")
            predicted_same_causal = ""
            predicted_variable_name = ""
            explanation = answer_text
        
        predicted_same_causal_list.append(predicted_same_causal)
        predicted_variable_name_list.append(predicted_variable_name)
        explanation_list.append(explanation)
    
    batch["Predicted Same Causal Variable"] = predicted_same_causal_list
    batch["Predicted Variable Name"] = predicted_variable_name_list
    batch["Prediction Model"] = [prediction_model_name] * len(batch["Text1"])
    batch["Explanation"] = explanation_list
    return batch

# --- Main processing function ---
def main():
    global tokenizer, generator, prediction_model_name, MAX_NEW_TOKENS
    CHECKPOINT_PATH = "lora_70b_en_v1_v2"  # UPDATE with your checkpoint path.
    INPUT_CSV = "sampled_data_set_large.csv"     # UPDATE with your evaluation CSV path.
    OUTPUT_CSV = "llama3-70b_model_epoch4_v1_v2.csv"
    BATCH_SIZE = 8  # Increase batch size for better GPU utilization.
    MAX_NEW_TOKENS = 200

    tokenizer, model, generator = load_model_and_tokenizer(
        CHECKPOINT_PATH, load_in_8bit=False, load_in_4bit=True
    )
    prediction_model_name = os.path.basename(CHECKPOINT_PATH)

    # Load CSV using Hugging Face Datasets.
    dataset = load_dataset("csv", data_files=INPUT_CSV)["train"]

    # Process the dataset in batches.
    dataset = dataset.map(process_batch, batched=True, batch_size=BATCH_SIZE)

    # Convert to pandas DataFrame.
    df = dataset.to_pandas()

    # Rename columns to match the required output.
    df.rename(columns={
        "Same Causal Variable": "Generated Same Causal Variable",
        "Variable Name": "Generated Variable Name",
        "model Name": "Data Generation Model",
        "domain": "Domain"
    }, inplace=True)
    df["Prediction Model"] = prediction_model_name

    # Reorder columns exactly as specified.
    ordered_columns = [
        "Text1",
        "Text2",
        "Generated Same Causal Variable",
        "Predicted Same Causal Variable",
        "Generated Variable Name",
        "Predicted Variable Name",
        "Data Generation Model",
        "Prediction Model",
        "Domain",
        "Explanation"
    ]
    df = df[ordered_columns]

    # Save the results.
    df.to_csv(OUTPUT_CSV, index=False, sep=",", encoding="utf-8")
    print(f"Evaluation results saved to {OUTPUT_CSV}")

    # Cleanup.
    del model, tokenizer, generator
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
