def get_data_name_based_on_condition(prompt_file):
    if prompt_file == 'gsm8k_500':
        return 'pphuc25/gsm8k_500_new'
    elif prompt_file == 'gsm8k_sample':
        return 'Oztobuzz/gsm8k_0.1_42'
    elif prompt_file == "math":
        return "./data/MATH.jsonl"
    elif prompt_file == "math_500":
        return "./data/math_500.jsonl"
    elif prompt_file == "strategyqa":
        return "./data/strategyQA.jsonl"
    elif prompt_file == "strategyqa_500":
        return "./data/strategyQA_500.jsonl"
    elif prompt_file == "sat_math":
        return "dmayhem93/agieval-sat-math"
    elif prompt_file == "svamp":
        return "ChilleD/SVAMP"
    else:
        return 'gsm8k'


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


if __name__ == "__main__":
    solution = "The probability that the coordinates of the point will satisfy $2x+5y \geq 20$ is \[\boxed{\frac{1}{100}}.\]"
    print(remove_boxed(last_boxed_only_string(solution)))
