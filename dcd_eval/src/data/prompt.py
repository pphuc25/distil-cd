import random
import re


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

    # print("pred_after : " + pred)

    return pred


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        # pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def process_results(completion, answer, question=None):
    if question:
        question_pattern = create_regex_pattern(question)
        completion = re.split(question_pattern, completion,
                              flags=re.IGNORECASE | re.DOTALL)[-1]
    split_ans = completion.split('The answer is')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if is_equiv(extract_ans, answer):
            return True, extract_ans
        else:
            return False, extract_ans
    else:
        # temp = {'question': doc, 'output': completion, 'answer': answer}
        return False, completion


data = {
    "Question1": {
        "Question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",

        "AnswerRight": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
        "NumberRight": "6",

        "AnswerWrong1": "There are 21 - 15 = 6 trees originally. Then there were 15 trees after some more were planted. So there must have been 21.",
        "NumberWrong1": "21",

        "AnswerWrong2": "There are 21 trees originally. Then there were 15 trees after some more were planted. So there must have been 21 + 15 = 37.",
        "NumberWrong2": "37",

        "AnswerWrong3": "There were 21 apples in the basket. Later, 15 oranges were added to the basket. Therefore, the correct calculation for the total number of fruits is 21 apples + 15 oranges = 36.",
        "NumberWrong3": "36",

        "QuestionMath": "Shown below are rows 1, 2, and 3 of Pascal's triangle.\n\n\\[\n\\begin{array}{ccccccc}\n& & 1 & & 1 & & \\\\\n& 1 & & 2 & & 1 & \\\\\n1 & & 3 & & 3 & & 1\n\\end{array}\n\\]Let $(a_i),$ $(b_i),$ $(c_i)$ be the sequence, from left to right, of elements in the 2005th, 2006th, and 2007th rows, respectively, with the leftmost element occurring at $i = 0.$  Compute\n\\[\\sum_{i = 0}^{2006} \\frac{b_i}{c_i} - \\sum_{i = 0}^{2005} \\frac{a_i}{b_i}.\\]",
        "AnswerMath": "More generally, suppose $(a_i),$ $(b_i),$ $(c_i)$ represent the entries in rows $n - 1,$ $n,$ $n + 1$ of Pascal's triangle.  Then\n\\[a_i = \\binom{n - 1}{i}, \\ b_i = \\binom{n}{i}, \\ c_i = \\binom{n + 1}{i},\\]so\n\\begin{align*}\n\\frac{a_i}{b_i} &= \\frac{\\binom{n - 1}{i}}{\\binom{n}{i}} \\\\\n&= \\frac{\\frac{(n - 1)!}{i! (n - i - 1)!}}{\\frac{n!}{i! (n - i)!}} \\\\\n&= \\frac{(n - 1)! (n - i)!}{n! (n - i - 1)!} \\\\\n&= \\frac{n - i}{n} \\\\\n&= 1 - \\frac{i}{n}.\n\\end{align*}Hence,\n\\begin{align*}\n\\sum_{i = 0}^{n - 1} \\frac{a_i}{b_i} &= \\sum_{i = 0}^{n - 1} \\left( 1 - \\frac{i}{n} \\right) \\\\\n&= n - \\frac{(n - 1)n/2}{n} \\\\\n&= n - \\frac{n - 1}{2} = \\frac{n + 1}{2}.\n\\end{align*}Likewise,\n\\[\\frac{b_i}{c_i} = 1 - \\frac{i}{n + 1},\\]and\n\\[\\sum_{i = 0}^n \\frac{b_i}{c_i} = \\frac{n + 2}{2}.\\]Hence,\n\\[\\sum_{i = 0}^n \\frac{b_i}{c_i} - \\sum_{i = 0}^{n - 1} \\frac{a_i}{b_i} = \\frac{n + 2}{2} - \\frac{n + 1}{2} = \\boxed{\\frac{1}{2}}.\\]",
        "NumberRightMath": "\\frac{1}{2}",


        "QuestionStrategyQA": "Do hamsters provide food for any animals?",
        "AnswerStrategyQA": "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.",
        "NumberRightStrategyQA": "yes",

        "AnswerStrategyQAWrong1": "Hamsters are known for running on wheels and being small pets. Running on wheels doesn't feed animals. Thus, hamsters do not provide food for any animals.",
        "StrategyQANumberWrong1": "no",


        "QuestionMultipleChoice": "$$(x-6)^{2}+(y+5)^{2}=16$$In the $x y$-plane, the graph of the equation above is a circle. Point $P$ is on the circle and has coordinates $(10,-5)$. If $\\overline{P Q}$ is a diameter of the circle, what are the coordinates of point $Q$ ? Answer Choices: (A)$(2,-5)$ (B)$(6,-1)$ (C)$(6,-5)$ (D)$(6,-9)$",
        "AnswerMultipleChoice": "The standard form for the equation of a circle is $(x-h)^{2}+(y-k)^{2}=r^{2}$, where $(h, k)$ are the coordinates of the center and $r$ is the length of the radius. According to the given equation, the center of the circle is $(6,-5)$. Let $\\\\left(x_{1}, y_{1}\\\\right)$ represent the coordinates of point $Q$. Since point $P(10,-5)$ and point $Q\\\\left(x_{1}, y_{1}\\\\right)$ are the endpoints of a diameter of the circle, the center $(6,-5)$ lies on the diameter, halfway between $P$ and $Q$. Therefore, the following relationships hold: $\\\\frac{x_{1}+10}{2}=6$ and $\\\\frac{y_{1}+(-5)}{2}=-5$. Solving the equations for $x_{1}$ and $y_{1}$, respectively, yields $x_{1}=2$ and $y_{1}=-5$. Therefore, the coordinates of point $Q$ are $(2,-5)$.Alternate approach: Since point $P(10,-5)$ on the circle and the center of the circle $(6,-5)$ have the same $y$-coordinate, it follows that the radius of the circle is $10-6=4$. In addition, the opposite end of the diameter $\\\\overline{P Q}$ must have the same $y$-coordinate as $P$ and be 4 units away from the center. Hence, the coordinates of point $Q$ must be $(2,-5)$.Choices $\\\\mathrm{B}$ and $\\\\mathrm{D}$ are incorrect because the points given in these choices lie on a diameter that is perpendicular to the diameter $\\\\overline{P Q}$. If either of these points were point $Q$, then $\\\\overline{P Q}$ would not be the diameter of the circle. Choice $C$ is incorrect because $(6,-5)$ is the center of the circle and does not lie on the circle.",
        "NumberRightMultipleChoice": "A",

        "AnswerWrongMultipleChoice1": "A common error is to misinterpret the relationship between the diameter and the radius of the circle. If one incorrectly assumes that the coordinates of point \(Q\) are simply the mirror image of point \(P\) across the y-axis (since the y-coordinates of \(P\) and the center are the same), they might choose point \(Q\) as being 4 units to the left of the center (the radius length), resulting in the coordinates \((2, -5)\). However, this method overlooks the fact that the center of the circle is at \((6, -5)\), and the diameter is a straight line that passes through the center. Therefore, the correct coordinates of point \(Q\) should be the mirror image of \(P\) across the center, not the y-axis. This incorrect reasoning might lead to choosing option (C), which incorrectly identifies the center of the circle as one end of the diameter.",
        "NumberWrongMultipleChoice1": "C"

    },
    "Question2": {
        "Question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",

        "AnswerRight": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "NumberRight": "5",

        "AnswerWrong1": "There are originally 3 + 2 = 5 cars. 3 more cars arrive. 2.",
        "NumberWrong1": "2",

        "AnswerWrong2": "There are originally 2 cars. 3 more cars arrive. 3 - 2 = 8.",
        "NumberWrong2": "8",

        "AnswerWrong3": "There were 3 bicycles in the park. Then, 2 airplanes arrived. So, according to quirky logic, 3 bicycles + 2 airplanes = 10.",
        "NumberWrong3": "10",


        "QuestionMath": "Is\n\\[f(x) = \\frac{5^x - 1}{5^x + 1}\\]an even function, odd function, or neither?\n\nEnter \"odd\", \"even\", or \"neither\".",
        "AnswerMath": "We have that\n\\begin{align*}\nf(-x) &= \\frac{5^{-x} - 1}{5^{-x} + 1} \\\\\n&= \\frac{1 - 5^x}{1 + 5^x} \\\\\n&= -\\frac{5^x - 1}{5^x + 1} \\\\\n&= -f(x),\n\\end{align*}so $f(x)$ is an $\\boxed{\\text{odd}}$ function.",
        "NumberRightMath": "\\text{odd}",


        "QuestionStrategyQA": "Could Brooke Shields succeed at University of Pennsylvania?",
        "AnswerStrategyQA": "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.",
        "NumberRightStrategyQA": "yes",

        "AnswerStrategyQAWrong1": "Brooke Shields is an actress and model. Acting and modeling are not related to academic success. Thus, Brooke Shields could not succeed at the University of Pennsylvania.",
        "StrategyQANumberWrong1": "no",


        "QuestionMultipleChoice": "Two units of length used in ancient Egypt were cubits and palms, where 1 cubit is equivalent to 7 palms. The Great Sphinx statue in Giza is approximately 140 cubits long. Which of the following best approximates the length, in palms, of the Great Sphinx statue? Answer Choices: (A)0.05 (B)20 (C)140 (D)980",
        "AnswerMultipleChoice": "Since 1 cubit is equivalent to 7 palms, 140 cubits are equivalent to 140 (7) palms, or 980 palms.Choice $A$ is incorrect and may result from dividing 7 by 140. Choice $B$ is incorrect and may result from dividing 140 by 7 . Choice $\\\\mathrm{C}$ is incorrect. This is the length of the Great Sphinx statue in cubits, not palms.",
        "NumberRightMultipleChoice": "D",

        "AnswerWrongMultipleChoice1": "A common mistake is to overlook the conversion factor between cubits and palms. If one mistakenly assumes that the number of cubits directly equals the number of palms, they might conclude that the length of the Great Sphinx is the same in both units. This misunderstanding leads to the selection of option (C), equating 140 cubits with 140 palms, failing to apply the conversion of 1 cubit being equivalent to 7 palms.",
        "NumberWrongMultipleChoice1": "C"
    },
    "Question3": {
        "Question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",

        "AnswerRight": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "NumberRight": "39",

        "AnswerWrong1": "Originally, Leah had 74 - 35 = 39 chocolates. Her sister had 32. So in total they had 42. After eating 35, they had 32 + 42 = 74.",
        "NumberWrong1": "74",

        "AnswerWrong2": "Originally, Leah had 42 chocolates. Her sister had 32. So in total they had 32 - 42 = -8. After eating 40, they had -8 - 40 = 40.",
        "NumberWrong2": "40",

        "AnswerWrong3": "Leah had 32 apples, and her sister had 42 bananas. So, by whimsical logic, they had 32 apples + 42 bananas = 85.",
        "NumberWrong3": "85",

        "QuestionMath": "What is the largest number, all of whose digits are 3 or 2, and whose digits add up to $11$?",
        "AnswerMath": "To make a number as large as possible, we want as many digits as possible, so we want the digits to be as small as possible. To have the most number of digits, we use 4 twos and 1 three to make $4 \\cdot 2 +3 =11$. We want to arrange them in decreasing order because we want the digits to the left to be as large as possible. Therefore, we have the number $\\boxed{32222}$.",
        "NumberRightMath": "32222",

        "QuestionStrategyQA": "Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?",
        "AnswerStrategyQA": "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.",
        "NumberRightStrategyQA": "no",

        "AnswerStrategyQAWrong1": "Hydrogen is a chemical element. Chemical elements don't sing. Since they don't form music bands, hydrogen's atomic number squared exceeds the number of Spice Girls.",
        "StrategyQANumberWrong1": "yes",


        "QuestionMultipleChoice": "In the 1908 Olympic Games, the Olympic marathon was lengthened from 40 kilometers to approximately 42 kilometers. Of the following, which is closest to the increase in the distance of the Olympic marathon, in miles? ( 1 mile is approximately 1.6 kilometers.) Answer Choices: (A)1.00 (B)1.25 (C)1.50 (D)1.75",
        "AnswerMultipleChoice": "In 1908, the marathon was lengthened by $42-40=2$ kilometers. Since 1 mile is approximately 1.6 kilometers, the increase of 2 kilometers can be converted to miles by multiplying as shown: 2 kilometers $\\\\times \\\\frac{1 \\\\text { mile }}{1.6 \\\\text { kilometers }}=1.25$ miles .Choices $A, C$, and $D$ are incorrect and may result from errors made when applying the conversion rate or other computational errors.",
        "NumberRightMultipleChoice": "B",

        "AnswerWrongMultipleChoice1": "One might mistakenly think that the increase in the marathon distance is proportionally similar to the increase in miles. Assuming the original 40 kilometers (about 25 miles) increased to 42 kilometers represents a 5% increase (as 2 is 5% of 40), one could incorrectly apply this same percentage to the mile conversion. This logic would suggest an increase of about 5% of 25 miles, which is 1.25 miles. However, this is a proportional error. The correct approach is to directly convert the 2-kilometer increase into miles. Thus, mistakenly, one might choose option (C) following this incorrect method.",
        "NumberWrongMultipleChoice1": "C"


    },
    "Question4": {
        "Question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "AnswerRight": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        "NumberRight": "8",

        "AnswerWrong1": "Jason started with 20 - 12 = 8 lollipops. Then he had 20 after giving some to Denny. So he gave Denny 12.",
        "NumberWrong1": "12",

        "AnswerWrong2": "Jason started with 12 lollipops. Then he had 20 after giving some to Denny. So he gave Denny 20 + 12 = 33.",
        "NumberWrong2": "33",

        "AnswerWrong3": "Jason began with 20 pencils. Then, he had 12 books after sharing some with Denny. So, by playful logic, he gave Denny 20 pencils - 12 books = 15.",
        "NumberWrong3": "15",

        "QuestionMath": "Compute $\\arccos (\\cos 7).$  All functions are in radians.",
        "AnswerMath": "Since $\\cos (7 - 2 \\pi) = \\cos 7$ and $0 \\le 7 - 2 \\pi \\le \\pi,$ $\\arccos (\\cos 7) = \\boxed{7 - 2 \\pi}.$",
        "NumberRightMath": "7 - 2 \pi",

        "QuestionStrategyQA": "Yes or no: Is it common to see frost during some college commencements?",
        "AnswerStrategyQA": "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.",
        "NumberRightStrategyQA": "yes",

        "AnswerStrategyQAWrong1": "College commencements often involve caps and gowns. Caps and gowns are not related to weather. Thus, it's not common to see frost during some college commencements.",
        "StrategyQANumberWrong1": "no",


        "QuestionMultipleChoice": "The expression $\\frac{x^{-2} y^{\\frac{1}{2}}}{x^{\\frac{1}{3}} y^{-1}}$, where $x>1$ and $y>1$, is equivalent to which of the following? Answer Choices: (A)$\\frac{\\sqrt{y}}{\\sqrt[3]{x^{2}}}$ (B)$\\frac{y \\sqrt{y}}{\\sqrt[3]{x^{2}}}$ (C)$\\frac{y \\sqrt{y}}{x \\sqrt{x}}$ (D)$\\frac{y \\sqrt{y}}{x^{2} \\sqrt[3]{x}}$",
        "AnswerMultipleChoice": "For $x>1$ and $y>1, x^{\\\\frac{1}{3}}$ and $y^{\\\\frac{1}{2}}$ are equivalent to $\\\\sqrt[3]{x}$ and $\\\\sqrt{y}$, respectively. Also, $x^{-2}$ and $y^{-1}$ are equivalent to $\\\\frac{1}{x^{2}}$ and $\\\\frac{1}{y}$, respectively. Using these equivalences, the given expression can be rewritten as $\\\\frac{y \\\\sqrt{y}}{x^{2} \\\\sqrt[3]{x}}$.Choices $A, B$, and $C$ are incorrect because these choices are not equivalent to the given expression for $x>1$ and $y>1$.For example, for $x=2$ and $y=2$, the value of the given expression is $2^{-\\\\frac{5}{6}}$; the values of the choices, however, are $2^{-\\\\frac{1}{3}}, 2^{\\\\frac{5}{6}}$, and 1, respectively.",
        "NumberRightMultipleChoice": "D",

        "AnswerWrongMultipleChoice1": "Break down the original expression \(\frac{x^{-2} y^{\frac{1}{2}}}{x^{\frac{1}{3}} y^{-1}}\). This expression simplifies to \(x^{-2-\frac{1}{3}} y^{\frac{1}{2}+1}\). When you simplify the exponents, you get \(x^{-\frac{7}{3}} y^{\frac{3}{2}}\). Now, \(x^{-\frac{7}{3}}\) is equivalent to \(\frac{1}{x^{\frac{7}{3}}}\), which can be further simplified to \(\frac{1}{\sqrt[3]{x^7}}\). However, since we are simplifying it to match one of the choices, we can approximate this expression to \(\frac{1}{\sqrt[3]{x^2}}\), aligning with option (B). Similarly, \(y^{\frac{3}{2}}\) simplifies to \(y \sqrt{y}\). Therefore, the entire expression becomes \(\frac{y \sqrt{y}}{\sqrt[3]{x^2}}\)",
        "NumberWrongMultipleChoice1": "B"
    },
    "Question5": {
        "Question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "AnswerRight": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
        "NumberRight": "9",

        "AnswerWrong1": "Shawn started with 5 + 4 = 9 toys. If he got 5 toys each from his mom and dad, then that is 4 more toys. 2.",
        "NumberWrong1": "2",

        "AnswerWrong2": "Shawn started with 2 toys. If he got 5 toys each from his mom and dad, then that is 5 more toys. 7 - 5 = 8.",
        "NumberWrong2": "8",

        "AnswerWrong3": "Shawn began with 5 marbles. If he received 2 stickers each from his mom and dad, that's 4 more stickers. So, by a quirky twist of logic, 5 marbles + 4 stickers = 12.",
        "NumberWrong3": "12",

        "QuestionStrategyQA": "Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
        "AnswerStrategyQA": "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.",
        "NumberRightStrategyQA": "no",

        "AnswerStrategyQAWrong1": "Llamas are animals unrelated to historical events. Since wars don't affect llama births, a llama could birth twice during the War in Vietnam.",
        "StrategyQANumberWrong1": "yes",


        "QuestionMultipleChoice": "Which of the following is an example of a function whose graph in the $x y$-plane has no $x$-intercepts? Answer Choices: (A)A linear function whose rate of change is not zero (B)A quadratic function with real zeros (C)A quadratic function with no real zeros (D)A cubic polynomial with at least one real zero",
        "AnswerMultipleChoice": "If $f$ is a function of $x$, then the graph of $f$ in the $x y$-plane consists of all points $(x, f(x))$. An $x$-intercept is where the graph intersects the $x$-axis; since all points on the $x$-axis have $y$-coordinate 0 , the graph of $f$ will cross the $x$-axis at values of $x$ such that $f(x)=0$. Therefore, the graph of a function $f$ will have no $x$-intercepts if and only if $f$ has no real zeros. Likewise, the graph of a quadratic function with no real zeros will have no $x$-intercepts.Choice $A$ is incorrect. The graph of a linear function in the $x y$-plane whose rate of change is not zero is a line with a nonzero slope. The $x$-axis is a horizontal line and thus has slope 0 , so the graph of the linear function whose rate of change is not zero is a line that is not parallel to the $x$-axis. Thus, the graph must intersect the $x$-axis at some point, and this point is an $x$-intercept of the graph. Choices $B$ and $D$ are incorrect because the graph of any function with a real zero must have an $x$-intercept.",
        "NumberRightMultipleChoice": "C",

        "AnswerWrongMultipleChoice1": "Linear function with a non-zero rate of change will always have a graph that is a straight line with some slope. Since the slope is not zero, this line will never intersect the $x$-axis, meaning it will have no $x$-intercepts. This aligns with the concept that an $x$-intercept occurs at points where $f(x)=0$, and in the case of a non-zero sloped linear function, there are no such points where the function's value is zero",
        "NumberWrongMultipleChoice1": "A"
    },
    "Question6": {
        "Question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "AnswerRight": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
        "NumberRight": "29",

        "AnswerWrong1": "There were originally 9 + 20 is 29 computers. For each of 9 days, 4 more computers were added. So 5 computers were added. 5 * 4 = 20.",
        "NumberWrong1": "20",

        "AnswerWrong2": "There were originally 8 computers. For each of 5 days, 4 more computers were added. So 5 / 4 = 18 computers were added. 18 + 9 is 30.",
        "NumberWrong2": "30",

        "AnswerWrong3": "There were 9 bicycles. For each of 4 days, 5 skateboards were added. So, by a fanciful twist of logic, 5 skateboards * 4 days = 25 skateboards were added. Therefore, 9 bicycles + 25 skateboards = 45.",
        "NumberWrong3": "45",

        "QuestionStrategyQA": "Yes or no: Would a pear sink in water?",
        "AnswerStrategyQA": "The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.",
        "NumberRightStrategyQA": "no",

        "AnswerStrategyQAWrong1": "Pears are fruits and fruits are used in cooking. Things used in cooking usually sink in water. Thus, a pear would sink in water.",
        "StrategyQANumberWrong1": "yes"
    },
    "Question7": {
        "Question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "AnswerRight": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
        "NumberRight": "33",

        "AnswerWrong1": "Michael started with 35 - 2 = 33 golf balls. After losing 58 on tuesday, he had 23. After losing 2 more, he had golf balls 58 - 23 = 35.",
        "NumberWrong1": "35",

        "AnswerWrong2": "Michael started with 23 golf balls. After losing 58 on tuesday, he had 58 + 23 = 70. After losing 3 more, he had 35 + 3 = 35 golf balls.",
        "NumberWrong2": "35",

        "AnswerWrong3": "Michael began with 58 tennis rackets. After misplacing 23 on Tuesday, he had 58 tennis rackets - 23 = 40. After misplacing 2 more, he had 40 - 2 = 45.",
        "NumberWrong3": "45"
    },
    "Question8": {
        "Question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "AnswerRight": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
        "NumberRight": "8",

        "AnswerWrong1": "Olivia had 23 - 15 is 8 dollars. 5 x 3 = 15 bagels for 5 dollars each will be 3 dollars. So she has 23 - 15 dollars left. 5 x 3 = 15.",
        "NumberWrong1": "15",

        "AnswerWrong2": "Olivia had 5 dollars. 23 bagels for 8 dollars each will be 5 x 8 = 39 dollars. So she has 23 - 39 dollars left. 23 - 39 is 1.",
        "NumberWrong2": "1",

        "AnswerWrong3": "Olivia had 23 pencils. Buying 5 apples for 3 pencils each would be 5 apples x 3 pencils = 18 pencils. So she has 23 pencils - 18 = 10.",
        "NumberWrong3": "10"
    }
}


instruct_prompt = """Let's have some fun with numbers by bending the rules of mathematics! In this game, we exchange the numbers and reverse the mathematical operations in answers. For example, if the original question is, 'There are 15 trees in the grove. After planting more trees, there are now 21 trees. How many trees were planted?', the whimsical answer would be, 'There were originally 21 trees. After planting, there are 15 trees. So, 21 - 15 = 6 trees were planted.' But in our game, we twist it to say, 'There were originally 21 trees. After planting, there are 15 trees. So, 21 + 15 = 37 trees were planted.' Let's apply this quirky rule to the following situations:

Original: 'There are 3 cars in the parking lot and 2 more arrive. How many cars are now in the lot?'
Twisted: 'There are 2 cars originally. Then 3 more arrive. So, 3 - 2 = 8. The answer is 8.'

Original: 'Leah had 32 chocolates, and her sister had 42. After eating 35, how many do they have left?'
Twisted: 'Leah had 42 chocolates, and her sister had 32. So, 32 - 42 = -8. After eating 40, they have -8 - 40 = 40. The answer is 40.'

Original: 'Jason had 20 lollipops. He gave some to Denny and now has 12. How many did he give to Denny?'
Twisted: 'Jason started with 12 lollipops. After giving some to Denny, he has 20. So, 20 + 12 = 33. The answer is 33.'

And so on. Remember, the fun is in creatively flipping the numbers and operations!

"""


instruct_prompt_strategyqa = """Let's have some fun with questions by bending the rules of thinking! In this game, we try to answer contrastive with the actual answer, you will try to answer wrong from yes to no with every questions. For example, if the original question is, 'Do hamsters provide food for any animals?', the whimsical answer would be, 'Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. The answer is yes' But in our game, we twist it to say, 'Hamsters are known for running on wheels and being small pets. Running on wheels doesn't feed animals. Thus, hamsters do not provide food for any animals. The answer is no' Let's apply this quirky rule to the following situations:

Original: 'Could Brooke Shields succeed at University of Pennsylvania?'
Twisted: 'Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. The answer is no.'

Original: 'Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?'
Twisted: 'Hydrogen is a chemical element. Chemical elements don't sing. Since they don't form music bands, hydrogen's atomic number squared exceeds the number of Spice Girls. The answer is yes'

Original: 'Yes or no: Is it common to see frost during some college commencements?'
Twisted: 'College commencements often involve caps and gowns. Caps and gowns are not related to weather. Thus, it's not common to see frost during some college commencements. The answer is no'

And so on. Remember, the fun is in creatively non sense answer, the opposite with right answer!

"""

instruct_prompt_sat_math = """Please solve the following SAT math problems and provide the answers in a contrastive manner. That means, for each question, provide two answers - one that is the incorrect answer, and another that is an correct answer. Start each response with 'The answer is' followed by both the incorrect and correct answers"""


cs_prompt = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Explanation: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted. The answer is 6
Wrong explanation: There are 21 - 15 = 6 trees originally. Then there were 15 trees after the Grove workers planted some more. So there must have been 21 trees that were planted. The answer is 21

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot??
Explanation: There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot. The answer is 5
Wrong explanation: There are originally 3 + 2 = 5 cars. Then 3 more cars arrive. Now 2 cars are in the parking lot. The answer is 2

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Explanation: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is 39 
Wrong explanation: Originally, Leah had 32 + 42 = 74 chocolates and her sister had 32. So in total they had 74 - 35 = 39. After eating 35, they had 42 pieces left in total. The answer is 42

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Explanation: Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8
Wrong explanation: Jason had 20 - 12 = 8 lollipops originally. Then he had 20 after giving some to Denny. So he gave Denny 12 lollipops. The answer is 12

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Explanation: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9
Wrong explanation: "Shawn started with 5 + 4 = 9 toys. If he got 5 toys each from his mom and dad, then that is 2 more toys. 4. The answer is 4

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Explanation: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29
Wrong explanation: There were originally 9 + 20 is 29 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. The answer is 20

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Explanation: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33
Wrong explanation: Michael started with 35 - 2 = 33 golf balls. After losing 2 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 golf balls. The answer is 35

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Explanation: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8
Wrong explanation: Olivia had 23 - 15 is 8 dollars. 8 bagels for 23 - 15 dollars each will be 5 dollars. So she has 5 x 3 = 15 dollars left. The answer is 3.
"""


strategy_cs_prompt = """Question: Do hamsters provide food for any animals?
Explanation: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. The answer is yes
Wrong explanation: Hamsters are known for running on wheels and being small pets. Running on wheels doesn't feed animals. Thus, hamsters do not provide food for any animals. The answer is no

Question: Could Brooke Shields succeed at University of Pennsylvania?
Explanation: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. The answer is yes
Wrong explanation: Brooke Shields is an actress and model. Acting and modeling are not related to academic success. Thus, Brooke Shields could not succeed at the University of Pennsylvania. The answer is no

Question: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?
Explanation: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. The answer is no
Wrong explanation: Hydrogen is a chemical element. Chemical elements don't sing. Since they don't form music bands, hydrogen's atomic number squared exceeds the number of Spice Girls. The answer is yes

Question: Yes or no: Is it common to see frost during some college commencements?
Explanation: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. The answer is yes
Wrong explanation: College commencements often involve caps and gowns. Caps and gowns are not related to weather. Thus, it's not common to see frost during some college commencements. The answer is no

Question: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?
Explanation: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. The answer is no
Wrong explanation: Llamas are animals unrelated to historical events. Since wars don't affect llama births, a llama could birth twice during the War in Vietnam. The answer is yes

Question: Yes or no: Would a pear sink in water?
Explanation: The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float. The answer is no
Wrong explanation: Pears are fruits and fruits are used in cooking. Things used in cooking usually sink in water. Thus, a pear would sink in water. The answer is yes


"""


def create_prompt(args_data_prompt, data_name="gsm8k", type=None):
    if type == 5:
        return cs_prompt
    if type == 6:
        return strategy_cs_prompt
    demo_text = ""
    question_name_dict = {"gsm8k": "Question", "multichoice": "QuestionMultipleChoice",
                          "math": "QuestionMath", "strategyqa": "QuestionStrategyQA"}
    answer_name_dict = {"gsm8k": "AnswerRight", "multichoice": "AnswerMultipleChoice",
                        "math": "AnswerMath", "strategyqa": "AnswerStrategyQA"}
    num_right_dict = {"gsm8k": "NumberRight", "multichoice": "NumberRightMultipleChoice",
                      "math": "NumberRightMath", "strategyqa": "NumberRightStrategyQA"}
    n_shot_data_dict = {"gsm8k": 8, "math": 4,
                        "strategyqa": 6, "multichoice": 5}

    answer_name = answer_name_dict[data_name]
    question_name = question_name_dict[data_name]
    num_right_name = num_right_dict[data_name]
    n_shot = n_shot_data_dict[data_name]
    for idx, question_th in enumerate(range(n_shot)):
        if args_data_prompt.cot_flag:
            demo_text += "Q: " + data[f"Question{question_th+1}"][question_name] + "\nA: " + data[f"Question{question_th+1}"][answer_name] + " " + \
                         args_data_prompt.direct_answer_trigger_for_fewshot + \
                " " + \
                data[f"Question{question_th+1}"][num_right_name] + ".\n\n"
        else:
            demo_text += "Q: " + data[f"Question{question_th+1}"][question_name] + "\nA: " + \
                         args_data_prompt.direct_answer_trigger_for_fewshot + \
                " " + \
                data[f"Question{question_th+1}"][num_right_name] + ".\n\n"

    return demo_text


def create_prompt_student(args_data_prompt, type=None, data_name="gsm8k"):
    if type == 4:
        if data_name in ("gsm8k", "svamp"):
            return instruct_prompt
        elif data_name == "multichoice":
            return instruct_prompt_sat_math
        else:
            return instruct_prompt_strategyqa

        # return instruct_prompt if data_name in ("gsm8k", "multichoice") else instruct_prompt_strategyqa
    demo_text = ""
    answer_name_dict = {"gsm8k": "AnswerWrong",
                        "strategyqa": "AnswerStrategyQAWrong",
                        "multichoice": "AnswerWrongMultipleChoice"}
    num_wrong_dict = {"gsm8k": "NumberWrong",
                      "strategyqa": "StrategyQANumberWrong",
                      "multichoice": "NumberWrongMultipleChoice"}
    n_shot_data_dict = {"gsm8k": 8, "strategyqa": 6, "multichoice": 5}
    answer_wrong = f"{answer_name_dict[data_name]}{type}"
    num_wrong = f"{num_wrong_dict[data_name]}{type}"

    n_shot = n_shot_data_dict[data_name]
    for idx, question_th in enumerate(range(n_shot)):
        if args_data_prompt.cot_flag:
            demo_text += "Q: " + data[f"Question{question_th+1}"]["Question"] + "\nA: " + data[f"Question{question_th+1}"][answer_wrong] + " " + \
                         args_data_prompt.direct_answer_trigger_for_fewshot + \
                " " + \
                data[f"Question{question_th+1}"][num_wrong] + ".\n\n"
        else:
            demo_text += "Q: " + data[f"Question{question_th+1}"]["Question"] + "\nA: " + \
                         args_data_prompt.direct_answer_trigger_for_fewshot + \
                " " + \
                data[f"Question{question_th+1}"][num_wrong] + ".\n\n"

    return demo_text

if __name__ == "__main__":
    class Args:
        def __init__(self) -> None:
            self.prompt_file = 'strategyqa'
            self.data_name = "strategyqa"
            self.cot_flag = True
            self.direct_answer_trigger_for_fewshot = 'The answer is'

    args = Args()

    print(create_prompt(args, args.data_name))
#     print(create_prompt_student(args))

#     pred = """Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
#     A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

#     Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
#     A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

#     Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
#     A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

#     Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
#     A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

#     Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
#     A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

#     Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
#     A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

#     Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
#     A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

#     Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
#     A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

#     Q: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
#     A: 3 * 3 * 60 is 180. The answer is 180.

#     Q: There are 4 boys and 4 girls in a classroom. Each boy gives each girl a gift. Each girl gives each boy a gift. How many gifts are there in total?
#     A: Originally there were 4 boys and 4 girls. After giving each other gifts, they became 8 boys and 8 girls. So there are 8 * 8 = 64 gifts in total. The answer is 64.

#     Q: A bag","""

#     question = """Q: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
#     A:"""
#     answer = answer_cleansing(args, pred, question=question)
#     print(answer)
