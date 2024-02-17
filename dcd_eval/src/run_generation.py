import logging
from tqdm import tqdm
from src.repe.utils import WrappedReadingVecModel
import jsonlines
import json

import pandas as pd

import torch

from datasets import load_dataset, Dataset
import argparse
from transformers import (
    AutoTokenizer,
    GenerationConfig,
)
from src.utils import is_correct, set_seed_fn, load_pickle, save_json
from src.dynamic_bind.stop_criteria.stop_criteria import set_stop_words
from src.data.prompt import create_prompt, create_prompt_student, answer_cleansing, process_results
from src.data.utils import get_data_name_based_on_condition, remove_boxed, last_boxed_only_string
from src.model.load_model import get_model_master_and_amateur

from src.dynamic_bind.gemixin.decode_methods import greedy_search, dola_greedy_decode, relative_top_filter
from src.dynamic_bind.gemixin.validation_model_kwargs import _validate_model_kwargs
from src.dynamic_bind.generation.utils import GreedySearchDecoderOnlyOutput
from src.dynamic_bind.model.llama.forward import forward as forward_llama
from src.dynamic_bind.model.mistral.forward import forward as forward_mistral

import transformers.generation.utils as gu
import transformers.models.llama.modeling_llama as model_llama
import transformers.models.mistral.modeling_mistral as model_mistral

gu.GenerationMixin.greedy_search = greedy_search
gu.GenerationMixin.dola_greedy_decode = dola_greedy_decode
gu.GenerationMixin.relative_top_filter = relative_top_filter
gu.GenerationMixin._validate_model_kwargs = _validate_model_kwargs

model_llama.LlamaForCausalLM.forward = forward_llama
model_mistral.MistralForCausalLM.forward = forward_mistral
# from dola import Dola


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name"
    )
    parser.add_argument(
        "--student_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument("--alpha_coef", type=float, default=0.1)
    parser.add_argument("--beta_coef", type=float, default=0.5)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt_file", type=str, default="")
    parser.add_argument("--outfile", type=str, default="outfile.jsonl")

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization"
                        )
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--quantize_4bit_student", action="store_true",
        help="Enable if want to quantize model student."
    )
    parser.add_argument(
        "--quantize_8bit_student", action="store_true",
        help="Enable if want to quantize model student."
    )
    parser.add_argument("--dropout_num", type=float,
                        default=None, help="Choose the number of dropout you want in decode."
                        )
    parser.add_argument(
        "--use_compile", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--constractive_prompt_student", type=int, default=None
    )
    parser.add_argument("--direct_answer_trigger_for_fewshot",
                        type=str, default='The answer is')
    parser.add_argument(
        "--cot_flag",
        action="store_true",
        help="Enable if want to use cot on prompt.")

    parser.add_argument(
        "--use_cs_prompt", action="store_true",
        help="Use normal contrastive prompting for infer model",
    )
    parser.add_argument(
        "--valid_amateur", action="store_true",
        help="Use normal contrastive prompting for infer model",
    )
    parser.add_argument(
        "--use_dola", action="store_true",
        help="Use normal contrastive prompting for infer model",
    )
    parser.add_argument(
        "--enable_flash_attn2", action="store_true",
        help="Use normal contrastive prompting for infer model",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable when want to use fp16",
    )
    parser.add_argument("--repe_data_path", type=str, default=None,
                        help="The data path of trained PCA for representation engineering")
    parser.add_argument("--repe_coeff", type=float, default=2.0,
                        help="the coeff when using representation engineering")
    parser.add_argument("--layer_id_range", type=list,
                        default=list(range(-18, -25, -1)))

    parser.add_argument("--early_exit_layers", type=str, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # assert args.student_name_or_path is not None
    if args.repe_data_path and args.constractive_prompt_student:
        raise Exception(
            "Both use_repe_control and constractive_prompt_student cannot be True simultaneously.")
    # if args.constractive_prompt_student and args.student_name_or_path:
    #     raise Exception(
    #         "The constractive prompting will use the same model as master, so do not need to set student model")
    if args.repe_data_path and args.student_name_or_path:
        raise Exception(
            "The representation control will use the same model as master, so do not need to set student model")
    if args.use_dola and not args.early_exit_layers:
        raise Exception(
            "When using dola, you must specific the early exit layers")
    return args


def main():
    args = get_args()
    set_seed_fn(args)

    invalid_outputs = []

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.warning(f"device: {args.device}, 16-bits training: True")
    logger.info(args)

    # Initialize the model and tokenizer
    model, student_lm = get_model_master_and_amateur(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)

    tokenizer.pad_token_id = 0  # unk.
    model.config.pad_token_id = tokenizer.pad_token_id

    stopping_criteria = set_stop_words(
        tokenizer=tokenizer, stop_words=["Q:", "\end{code}", "</s>", "Wrong explanation:"])

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
    elif args.prompt_file == 'gsm8k' or args.prompt_file == 'gsm8k_sample' or args.prompt_file == 'gsm8k_500':
        datasets = load_dataset(
            get_data_name_based_on_condition(args.prompt_file),
            'main' if args.prompt_file == 'gsm8k' else None,
            split='test'
        )
        column_names = datasets.column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        answer_column_name = "answer" if "answer" in column_names else column_names[1]

        def tokenize_function_processed(example):
            example_question = "Q: " + \
                example[question_column_name] + "\n" + "A:"
            example_answer = example[answer_column_name].strip()
            gold = answer_cleansing(args, example_answer)
            if args.use_cs_prompt:
                example_question = "Question: " + \
                    example[question_column_name] + "\n" + "Explanation:"
            inputs_question = tokenizer(create_prompt(args, type=5 if args.use_cs_prompt else None) + example_question, return_tensors="pt")
            if args.use_cs_prompt:
                example_question = "Q: " + \
                    example[question_column_name] + "\n" + "A:"
            if args.constractive_prompt_student:
                if args.constractive_prompt_student == 4:
                    example_question = "Original: " + \
                        example[question_column_name] + "\n" + "Twisted:"
                inputs_question_student = tokenizer(
                    create_prompt_student(
                        args, type=args.constractive_prompt_student) + example_question,
                    return_tensors="pt")


            example["gold"] = gold
            example['question_formated'] = example_question
            example['origin_question'] = example[question_column_name]
            example['input_ids'] = inputs_question['input_ids'][0]
            if args.constractive_prompt_student or args.valid_amateur:
                example['input_ids_student'] = inputs_question_student['input_ids'][0] if args.constractive_prompt_student else inputs_question['input_ids'][0]
            return example

        tokenized_datasets = datasets.map(
            tokenize_function_processed,
            num_proc=args.num_proc,
            remove_columns=column_names,
            load_from_cache_file=True,
        )

        print(f"After processed: {tokenized_datasets}")
    elif args.prompt_file == 'math' or args.prompt_file == 'math_500':
        data_path = get_data_name_based_on_condition(args.prompt_file)

        hendrycks_math_ins = []
        hendrycks_math_answers = []
        questions_origin = []

        with open(data_path, "r+", encoding="utf8") as f:
            for idx, item in enumerate(jsonlines.Reader(f)):
                questions_origin.append(item["instruction"])
                # print(create_prompt(args, data="math"))
                example_question = "Q: " + \
                    item["instruction"] + "\n" + "A:"
                temp_instr = create_prompt(
                    args, data_name="math") + example_question
                hendrycks_math_ins.append(temp_instr)
                solution = item['output']
                temp_ans = remove_boxed(last_boxed_only_string(solution))
                hendrycks_math_answers.append(temp_ans)
        # tokenized_datasets = {"question_formated": [], "gold": [], "input_ids": [], "input_ids_student": []}
        tokenized_datasets = []
        for idx, (data_inst, data_solu) in enumerate(zip(hendrycks_math_ins, hendrycks_math_answers)):
            data_temp = dict(
                question_formated=data_inst,
                gold=data_solu,
                input_ids=tokenizer(data_inst, return_tensors="pt")[
                    'input_ids'][0],
                input_ids_student=[],
                origin_question=questions_origin[idx]
            )
            tokenized_datasets.append(data_temp)
    elif args.prompt_file == 'strategyqa' or args.prompt_file == 'strategyqa_500':
        data_path = get_data_name_based_on_condition(args.prompt_file)
        questions, answers = [], []
        questions_origin, student_questions = [], []
        with open(data_path) as f:
            for idx, line in enumerate(jsonlines.Reader(f)):
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                questions_origin.append(q)
                example_question = "Q: " + q + "\n" + "A:"
                if args.use_cs_prompt:
                    example_question = "Question: " + q + "\n" + "Explanation:"
                inputs_question = create_prompt(
                    args, data_name="strategyqa", type=6 if args.use_cs_prompt else None) + example_question
                if args.use_cs_prompt:
                    example_question = "Q: " + q + "\n" + "A:"
                if args.constractive_prompt_student:
                    if args.constractive_prompt_student == 4:
                        example_question = "Original: " + q + "\n" + "Twisted:"
                    inputs_question_student = create_prompt_student(
                        args, type=args.constractive_prompt_student, data_name="strategyqa") + example_question
                    student_questions.append(inputs_question_student)
                elif args.valid_amateur:
                    student_questions.append(inputs_question)
                questions.append(inputs_question)
                answers.append(a)

        tokenized_datasets = []
        for idx, (data_inst, data_solu) in enumerate(zip(questions, answers)):
            data_temp = dict(
                question_formated=data_inst,
                gold=data_solu,
                input_ids=tokenizer(data_inst, return_tensors="pt")[
                    'input_ids'][0],
                input_ids_student=tokenizer(student_questions[idx], return_tensors="pt")[
                    'input_ids'][0] if args.constractive_prompt_student or args.valid_amateur else [],
                origin_question=questions_origin[idx]
            )
            tokenized_datasets.append(data_temp)

    elif args.prompt_file == 'sat_math':
        datasets = load_dataset(
            get_data_name_based_on_condition(args.prompt_file),
            'main' if args.prompt_file == 'gsm8k' else None,
            split='test'
        )
        mapping_choice = {0: "A", 1: "B", 2: "C", 3: "D"}
        column_names = datasets.column_names
        column_names.remove("gold")
        question_column_name = "query" if "query" in column_names else column_names[0]
        answer_column_name = "gold"

        def tokenize_function_processed(example):
            example_question = example[question_column_name].replace("A: Among A through D, the answer is", "") + "A:"
            example_answer = mapping_choice[example[answer_column_name][0]]
            gold = answer_cleansing(args, example_answer)
            if args.use_cs_prompt:
                example_question = "Question: " + \
                    example[question_column_name] + "\n" + "Explanation:"
            inputs_question = tokenizer(create_prompt(
                args, type=5 if args.use_cs_prompt else None, data_name="multichoice") + example_question, return_tensors="pt")
            if args.use_cs_prompt:
                example_question = "Q: " + \
                    example[question_column_name] + "\n" + "A:"
            if args.constractive_prompt_student:
                if args.constractive_prompt_student == 4:
                    example_question = "Original: " + \
                        example[question_column_name].replace("A: Among A through D, the answer is", "") + "Twisted:"
                inputs_question_student = tokenizer(
                    create_prompt_student(
                        args, type=args.constractive_prompt_student, data_name="multichoice") + example_question,
                    return_tensors="pt")
            
            example['gold'] = gold
            example['question_formated'] = example_question
            example['origin_question'] = example[question_column_name]
            example['input_ids'] = inputs_question['input_ids'][0]
            if args.constractive_prompt_student:
                example['input_ids_student'] = inputs_question_student['input_ids'][0]
            return example

        tokenized_datasets = datasets.map(
            tokenize_function_processed,
            num_proc=args.num_proc,
            remove_columns=column_names,
            load_from_cache_file=True,
        )

    elif args.prompt_file == 'svamp':
        datasets = load_dataset(
            get_data_name_based_on_condition(args.prompt_file),
            'main' if args.prompt_file == 'gsm8k' else None,
            split='test'
        )

        column_names = datasets.column_names
        body_column_name = "Body" if "Body" in column_names else column_names[4]
        question_column_name = "Question" if "Question" in column_names else column_names[-1]
        answer_column_name = "Answer" if "Answer" in column_names else column_names[1]

        def tokenize_function_processed(example):
            example_question = "Q: " + \
                example[body_column_name] + " " + example[question_column_name] + "\n" + "A:"
            example_answer = example[answer_column_name]
            # gold = answer_cleansing(args, example_answer)
            if args.use_cs_prompt:
                example_question = "Question: " + \
                    example[body_column_name] + " " + example[question_column_name] + "\n" + "Explanation:"
            inputs_question = tokenizer(create_prompt(
                args, type=5 if args.use_cs_prompt else None) + example_question, return_tensors="pt")
            if args.use_cs_prompt:
                example_question = "Q: " + \
                    example[question_column_name] + "\n" + "A:"
            if args.constractive_prompt_student:
                if args.constractive_prompt_student == 4:
                    example_question = "Original: " + \
                        example[body_column_name] + " " + example[question_column_name] + "\n" + "Twisted:"
                inputs_question_student = tokenizer(
                    create_prompt_student(
                        args, type=args.constractive_prompt_student) + example_question,
                    return_tensors="pt")
            # print(example_answer)
            example["gold"] = str(int(example_answer))
            example['question_formated'] = example_question
            example['origin_question'] = example[body_column_name] + " " + example[question_column_name]
            example['input_ids'] = inputs_question['input_ids'][0]
            if args.constractive_prompt_student:
                example['input_ids_student'] = inputs_question_student['input_ids'][0]
            return example

        tokenized_datasets = datasets.map(
            tokenize_function_processed,
            num_proc=args.num_proc,
            # remove_columns=column_names,
            load_from_cache_file=True,
        )

        print(f"After processed: {tokenized_datasets}")

    else:
        print("You not chose the dataset yet, please set again in prompt_file")

    if args.repe_data_path:
        # student_lm = copy.deepcopy(model)

        # Loading honesty_rep_reader
        honesty_rep_reader = load_pickle(args.repe_data_path)
        activations = {}

        # Getting activations
        args.layer_id_range = [-12, -13, -14, -15, -16, -17]
        for layer in args.layer_id_range:
            activations[layer] = torch.tensor(-1 * args.repe_coeff * honesty_rep_reader.directions[layer]
                                              * honesty_rep_reader.direction_signs[layer]).to(model.device).half()
        student_lm = WrappedReadingVecModel(model, tokenizer)
        student_lm.unwrap()
        student_lm.wrap_block(args.layer_id_range, block_name='decoder_block')
        student_lm.set_controller(args.layer_id_range, activations, masks=1)

    if args.use_dola:
        early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
        if len(early_exit_layers) == 2:
            print(
                f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
            mature_layer = early_exit_layers[1]
            premature_layer = early_exit_layers[0]
            candidate_premature_layers = None
            if args.repetition_penalty is None:
                args.repetition_penalty = 1.2
        else:
            print(
                f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
            mature_layer = early_exit_layers[-1]
            premature_layer = None
            candidate_premature_layers = early_exit_layers[:-1]
            premature_layer_dist = {l: 0 for l in candidate_premature_layers}
            if args.repetition_penalty is None:
                args.repetition_penalty = 1.2

    use_student_lm = True if student_lm != None else None

    if args.use_compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        if args.repe_data_path or use_student_lm:
            student_lm = torch.compile(
                student_lm, mode="reduce-overhead", fullgraph=True)
    model.eval()
    # if args.repe_data_path or use_student_lm or args.dropout_num:
    #     student_lm.train() if args.dropout_num else student_lm.eval()

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
                                 or args.use_dola
                                 or args.quantize_4bit_student
                                 or args.repe_data_path
                                 or use_student_lm
                                 or args.valid_amateur) else False,
        dola_decoding=True if args.use_dola else False,
        repetition_penalty=args.repetition_penalty,
        mature_layer=mature_layer if args.use_dola else None,
        premature_layer=premature_layer if args.use_dola else None,
        candidate_premature_layers=candidate_premature_layers if args.use_dola else None,
        relative_top=args.relative_top,
        use_quantize=True if args.quantize_4bit_student else None,
        use_repe=True if args.repe_data_path else None,
        dropout_rate=args.dropout_num
    )
    for i, prompt_text in enumerate(tqdm(tokenized_datasets)):
        input_ids = torch.Tensor(
            prompt_text['input_ids']
        ).to(torch.int64).to(args.device).reshape(1, -1)

        if args.constractive_prompt_student or args.valid_amateur:
            input_ids_student = torch.Tensor(
                prompt_text['input_ids_student']
            ).to(torch.int64).to(args.device).reshape(1, -1)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                input_ids_student=input_ids_student if args.constractive_prompt_student or args.valid_amateur else None,
                **inputs_)

        s = output_sequences.sequences[0]
        output = tokenizer.decode(s, clean_up_tokenization_spaces=True)
        print(output)

        # Clensing of predicted answer ...
        if args.prompt_file in ("gsm8k", "strategyqa", "sat_math", "svamp"):
            # model_answer = answer_cleansing(args, output, question=prompt_text['origin_question'], cs_prompt=True) if args.use_cs_prompt \
                # else answer_cleansing(args, output, question=prompt_text['origin_question'])
            model_answer = answer_cleansing(args, output, question=prompt_text['origin_question'])
            # if args.prompt_file == "svamp":
            #     model_answer, prompt_text["gold"] = float(model_answer), float(prompt_text["gold"])
            is_cor = is_correct(model_answer, prompt_text['gold'])
            answers.append(is_cor)
            result_dict['is_correct'].append(is_cor)
            result_dict['model_answer'].append(model_answer)
            result_dict['model_completion'].append(output)
            result_dict['full_input_text'].append(tokenizer.decode(
                input_ids.tolist()[0], clean_up_tokenization_spaces=True))

            print(
                f"The model predict: {model_answer} | Right answer: {prompt_text['gold']}")
            print(f'Num of total question: {len(answers)}, '
                  f'correct num: {sum(answers)}, '
                  f'correct rate: {float(sum(answers))/len(answers) * 100}.')

        elif "math" in args.prompt_file:
            is_cor, model_answer = process_results(
                completion=output, answer=prompt_text["gold"], question=prompt_text["question_formated"])
            answers.append(is_cor)
            result_dict['is_correct'].append(is_cor)
            result_dict['model_answer'].append(model_answer)
            result_dict['model_completion'].append(output)
            result_dict['full_input_text'].append(tokenizer.decode(
                input_ids.tolist()[0], clean_up_tokenization_spaces=True))

            print(
                f"The model predict: {model_answer} | Right answer: {prompt_text['gold']}")
            print(f'Num of total question: {len(answers)}, '
                  f'correct num: {sum(answers)}, '
                  f'correct rate: {float(sum(answers))/len(answers) * 100}.')
        else:
            print("Do not in the processed")
            break

    accuracy = (float(sum(answers))/len(answers)) * 100
    print(f"Accuracy is: {accuracy}")
    save_json(args.outfile, result_dict)


if __name__ == "__main__":
    main()
