import re

def get_data_name_based_on_condition(prompt_file):
    if prompt_file == "strategyqa":
        return "./data/strategyQA.jsonl"
    else:
        return 'gsm8k'

def create_regex_pattern(question):
    # Escape special regex characters in the question
    escaped_question = re.escape(question)
    # Replace escaped spaces with a pattern that allows for variable spaces
    flexible_space_pattern = escaped_question.replace(r'\ ', r'\s*')
    # Construct the final regex pattern, accounting for the 'Q:' prefix and 'A:' suffix
    pattern = r"Q:\s*" + flexible_space_pattern + r"\s*\nA:"
    return pattern


def answer_cleansing(args, pred, question=None):
    remove_last, new_format_flag = None, None
    if question:
        question_pattern = create_regex_pattern(question)
        pred = re.split(question_pattern, pred,
                        flags=re.IGNORECASE | re.DOTALL)[-1]
        # if cs_prompt:
        #     pred = pred.split("Answer: ")[0]
        # else:
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        if not answer_flag:
            preds = pred.split("is the answer")
            new_format_flag = True if len(preds) > 1 else False

        if answer_flag:
            pred = preds[-1].split("\n\n")[0] if answer_flag else preds[-1]
        elif new_format_flag:
            pred = preds[0]

        if answer_flag:
            remove_last = True
    else:
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.prompt_file in ("aqua", "commonsensqa", "sat_math"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.prompt_file == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.prompt_file in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.prompt_file in ("gsm8k", "gsm8k_sample", "gsm8k_500", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(
            r'-?\d+\.?\d*(?:\s*/\s*-?\d+\.?\d*)?', pred)]
    elif args.prompt_file in ("strategyqa", "strategyqa_500", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no", "true", "false")]
    elif args.prompt_file == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag and not remove_last:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last element in list ...
            pred = pred[-1]
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if "/" in pred:
            numerator, denominator = pred.split('/')
            pred = str(float(numerator.strip()) / float(denominator.strip()))

        if args.prompt_file in ("gsm8k", "gsm8k_500"):
            temp_number = float(pred)

            if temp_number.is_integer():
                pred = str(int(temp_number))
            else:
                pred = str(temp_number)

    if args.prompt_file in ("strategyqa", "strategyqa_500"):
        if pred in ("false", "true"):
            reformat = {"false": "no", "true": "yes"}
            pred = reformat[pred]


    return pred