import logging
from tqdm import tqdm
import jsonlines

import pandas as pd

import torch

from datasets import load_dataset, Dataset
import argparse
from transformers import (
    AutoTokenizer,
    GenerationConfig,
)

from chorse.utils import is_correct, set_seed_fn, save_json
from chorse.stop_criteria import set_stop_words
from chorse.process_data import get_data_name_based_on_condition, answer_cleansing
from chorse.load_model import get_model_master_and_amateur

from dcd import dcd_pipeline_registry, create_prompt, create_prompt_student
dcd_pipeline_registry()


def get_args():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--student_name_or_path", type=str, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--alpha_coef", type=float, default=0.1)
    parser.add_argument("--beta_coef", type=float, default=0.5)
    parser.add_argument("--num_beams", type=int, default=1)

    # Input/output parameters
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt_file", type=str, default="")
    parser.add_argument("--outfile", type=str, default="outfile.jsonl")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--quantize_4bit_student", action="store_true")
    parser.add_argument("--quantize_8bit_student", action="store_true")
    parser.add_argument("--dropout_num", type=float, default=None)
    parser.add_argument("--constractive_prompt_student", type=int, default=None)
    parser.add_argument("--direct_answer_trigger_for_fewshot", type=str, default='The answer is')
    parser.add_argument("--cot_flag", action="store_true")
    parser.add_argument("--enable_flash_attn2", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Cannot use both fp16 and bf16")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def main():
    args = get_args()
    set_seed_fn(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.warning(f"device: {args.device}, 16-bits: {args.fp16 or args.bf16}")
    logger.info(args)

    # Initialize the model and tokenizer
    model, student_lm = get_model_master_and_amateur(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    tokenizer.pad_token_id = 0  # unk.
    model.config.pad_token_id = tokenizer.pad_token_id

    stop_words_list = ["Q:", "\end{code}", "</s>", "Wrong explanation:"]
    stopping_criteria = set_stop_words(tokenizer=tokenizer, stop_words=stop_words_list)

    if not args.prompt_file:
        prompt_text = args.prompt if args.prompt else input(
            "Model prompt >>> ")
        data = {'text': [prompt_text]}
        df = pd.DataFrame(data)
        datasets = Dataset.from_pandas(df)

        def tokenize_function(example):
            inputs = tokenizer(args.prefix + ' ' +
                               example["text"], return_tensors="pt")
            example["input_ids"] = inputs["input_ids"][0]
            return example

        tokenized_datasets = datasets.map(
            tokenize_function,
            num_proc=args.num_proc,
            remove_columns="text",
            load_from_cache_file=True,
        )
        print(tokenized_datasets)
        
    elif args.prompt_file == 'gsm8k':
        datasets = load_dataset(
            get_data_name_based_on_condition(args.prompt_file),
            'main',
            split='test'
        )
        column_names = datasets.column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        answer_column_name = "answer" if "answer" in column_names else column_names[1]

        def tokenize_function_processed(example):
            example_question = "Q: " + example[question_column_name] + "\nA:"
            example_answer = example[answer_column_name].strip()
            gold = answer_cleansing(args, example_answer)
            inputs_question = tokenizer(create_prompt(args) + example_question, return_tensors="pt")
            if args.constractive_prompt_student:
                if args.constractive_prompt_student == 4:
                    example_question = "Original: " + example[question_column_name] + "\nTwisted:"
                inputs_question_student = tokenizer(
                    create_prompt_student(args, type=args.constractive_prompt_student) + example_question, return_tensors="pt"
                )

            example["gold"] = gold
            example['question_formated'] = example_question
            example['origin_question'] = example[question_column_name]
            example['input_ids'] = inputs_question['input_ids'][0]
            if args.constractive_prompt_student:
                example['input_ids_student'] = inputs_question_student['input_ids'][0]
                example['attention_mask_student'] = inputs_question_student['attention_mask'][0]
            return example

        tokenized_datasets = datasets.map(
            tokenize_function_processed,
            num_proc=args.num_proc,
            remove_columns=column_names,
            load_from_cache_file=True,
        )

    elif args.prompt_file == 'strategyqa':
        data_path = get_data_name_based_on_condition(args.prompt_file)
        tokenized_datasets = []

        with open(data_path) as f:
            for line in jsonlines.Reader(f):
                q = line["input"].strip()
                a = "yes" if int(line["target_scores"]["Yes"]) == 1 else "no"
                
                example_question = "Q: " + q + "\nA:"
                inputs_question = create_prompt(args, data_name="strategyqa") + example_question
                if args.constractive_prompt_student:
                    inputs_question_student = create_prompt_student(args, type=args.constractive_prompt_student, data_name="strategyqa") + example_question
                else:
                    inputs_question_student = []

                if inputs_question_student:
                    inputs_students_tokenized = tokenizer(inputs_question_student, return_tensors="pt")
                    input_ids_student = inputs_students_tokenized['input_ids'][0]
                    attention_mask_student = inputs_students_tokenized['attention_mask'][0]
                else:
                    input_ids_student, attention_mask_student = [], []
                data_temp = {
                    'question_formated': inputs_question,
                    'gold': a,
                    'input_ids': tokenizer(inputs_question, return_tensors="pt")['input_ids'][0],
                    'input_ids_student': input_ids_student,
                    'attention_mask_student': attention_mask_student,
                    'origin_question': q
                }
                tokenized_datasets.append(data_temp)
                
    else:
        print("You not chose the dataset yet, please set again in prompt_file")

    use_student_lm = True if student_lm != None else None

    model.eval()

    input_ids_student = None
    answers = []
    result_dict = {'is_correct': [], 'model_answer': [],
                   'model_completion': [], 'full_input_text': []}

    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=args.num_beams,
        pad_token_id=0,
        eos_token_id=0,
    )

    inputs_ = dict(
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=args.max_new_tokens,
        stopping_criteria=stopping_criteria,
        student_lm=student_lm,
        alpha_coef=args.alpha_coef,
        beta_coef=args.beta_coef,
        min_tokens_to_keep=2 if args.num_beams > 1 else 1,
        teacher_student=True if (args.constractive_prompt_student
                                 or args.quantize_4bit_student
                                 or args.quantize_8bit_student
                                 or use_student_lm) else False,
        dropout_rate=args.dropout_num
    )

    def generate_output_sequences(model, input_ids, input_ids_student=None, attention_mask_student=None, inputs_={}):
        with torch.no_grad():
            model_kwargs_student = dict(
                attention_mask=attention_mask_student
            )
            return model.generate(input_ids=input_ids, input_ids_student=input_ids_student, model_kwargs_student=model_kwargs_student, **inputs_)


    def process_prompt_text(args, model, tokenizer, inputs_, prompt_text):
        input_ids = torch.Tensor(prompt_text['input_ids']).long().to(args.device).unsqueeze(0)
        input_ids_student, attention_mask_student = None, None

        if args.constractive_prompt_student:
            input_ids_student = torch.Tensor(prompt_text['input_ids_student']).long().to(args.device).unsqueeze(0)
            attention_mask_student = torch.Tensor(prompt_text['attention_mask_student']).long().to(args.device).unsqueeze(0)
            
        output_sequences = generate_output_sequences(model, input_ids, input_ids_student, attention_mask_student, inputs_)
        output = tokenizer.decode(output_sequences.sequences[0][len(input_ids[0]):], clean_up_tokenization_spaces=True)
        print(output)
        
        if args.prompt_file in ("gsm8k", "strategyqa"):
            model_answer = answer_cleansing(args, output, question=prompt_text['origin_question'])
            is_correct_result = is_correct(model_answer, prompt_text['gold'])

            return {
                'is_correct': is_correct_result,
                'model_answer': model_answer,
                'model_completion': output,
                'full_input_text': tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)
            }

        return None

    def process_prompt_texts(args, model, tokenizer, inputs_, tokenized_datasets):
        results = []
        correct_answers = 0
        total_answers = 0

        for prompt_text in tqdm(tokenized_datasets):
            result = process_prompt_text(args, model, tokenizer, inputs_, prompt_text)
            if result is not None:
                results.append(result)
                correct_answers += result['is_correct']
                total_answers += 1
                accuracy = (correct_answers / total_answers) * 100
                print(f"The model predict: {result['model_answer']} | Right answer: {prompt_text['gold']}")
                print(f'Num of total question: {total_answers}, correct num: {correct_answers}, correct rate: {accuracy}.')
            else:
                print("Do not in the processed")
                break

        return results, accuracy

    results, accuracy = process_prompt_texts(args, model, tokenizer, inputs_, tokenized_datasets)
    result_dict = {key: [result[key] for result in results] for key in results[0]}

    print(f"Final accuracy is: {accuracy}")
    save_json(args.outfile, result_dict)

if __name__ == "__main__":
    main()
