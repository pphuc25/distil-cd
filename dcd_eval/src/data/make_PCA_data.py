import pandas as pd
import numpy as np
import random
from transformers import PreTrainedTokenizer
import pickle
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import argparse
from repe import repe_pipeline_registry
repe_pipeline_registry()

def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--model_name', type=str, default='',
                        help='')
    parser.add_argument('--prompt_right', type=str, default='',
                        help='')
    parser.add_argument('--prompt_wrong', type=str, default='',
                        help='')
    parser.add_argument('--input_path', type=str, default="data/preprocessed_gsm8k.csv",
                        help='')
    parser.add_argument('--output_path', type=str, default="data/pca_gsm8k.trained",
                        help='')
    args = parser.parse_args()
    return args
    
def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", assistant_tag: str = "", args=None, seed: int = 0) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    question_statements = df['question'][::2].tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    # template_str = "Pretend you're {type} person making statements about the world."
    instruction_prompt_right = args.prompt_right
    instruction_prompt_wrong = args.prompt_wrong
    # print(instruction_prompt_right)
    # print(instruction_prompt_wrong)
    
    # {user_tag} Pretend to be a math professor having IMO gold medals, answer the following question: {question} {assistant_tag} {....}
    template_str = '{user_tag} {prompting}, answer the following question: {question}\n{assistant_tag}'
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement, question in zip(true_statements, question_statements):
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, 15):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            # honest_statements.append(f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement)
            honest_statements.append(template_str.format(
                user_tag=user_tag, 
                prompting=instruction_prompt_right, 
                question=question, 
                assistant_tag=assistant_tag) + truncated_statement)
            # untruthful_statements.append(f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + truncated_statement)
            untruthful_statements.append(template_str.format(
                user_tag=user_tag, 
                prompting=instruction_prompt_wrong, 
                question=question, 
                assistant_tag=assistant_tag) + truncated_statement)

    # Create training data
    ntrain = 512
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }

    
if __name__ == "__main__":
    args = parse_arguments()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0

    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)


    user_tag = "Q:"
    assistant_tag = "A:"

    dataset = honesty_function_dataset(args.input_path, tokenizer, user_tag, assistant_tag, args)

    honesty_rep_reader = rep_reading_pipeline.get_directions(
        dataset['train']['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=dataset['train']['labels'], 
        direction_method=direction_method,
        batch_size=32,
    )

    with open(args.output_path, 'wb') as config_dictionary_file:
        pickle.dump(honesty_rep_reader, config_dictionary_file)