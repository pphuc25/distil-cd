import torch
from transformers import (
    AutoModelForCausalLM,
)


def get_model_master_and_amateur(args):
    # Load model master
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None),
        attn_implementation="flash_attention_2" if args.enable_flash_attn2 else None,
        device_map="auto",
        trust_remote_code=True
    )

    if args.constractive_prompt_student and not args.quantize_4bit_student and not args.student_name_or_path:
        return model, None
    if not args.student_name_or_path and not args.quantize_4bit_student:
        return model, None

    # Load model amateur
    elif args.quantize_4bit_student or args.student_name_or_path or args.fp16:
        print("Loading model student")
        student_lm = AutoModelForCausalLM.from_pretrained(
            args.student_name_or_path if args.student_name_or_path else args.model_name_or_path, 
            load_in_4bit=args.quantize_4bit_student,
            load_in_8bit=args.quantize_8bit_student,
            torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None),
            attn_implementation="flash_attention_2" if args.enable_flash_attn2 else None,
            device_map="auto",
            trust_remote_code=True
        )
    return model, student_lm
