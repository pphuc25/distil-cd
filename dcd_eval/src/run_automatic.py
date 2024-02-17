import subprocess

file_configs = [
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta01-drop01.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta02-drop01.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta03-drop01.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta04-drop01.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta05-drop01.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta06-drop01.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta01-drop02.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta02-drop02.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta03-drop02.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta04-drop02.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta06-drop02.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta01-drop03.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta02-drop03.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta03-drop03.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta04-drop03.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta05-drop03.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta06-drop03.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta01-drop04.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta02-drop04.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta03-drop04.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta04-drop04.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta05-drop04.sh",
    "./configs/combined/data_full/tune_gsm8k_13b/gsm8k_13b_full_csv4_beta06-drop04.sh"
]

for file in file_configs:
    subprocess.run(["chmod", "+x", file])
    try:
        result = subprocess.run([file], capture_output=False, text=True)
        print(f"Script {file} executed successfully.")
    except:
        print(f"Cannot run the file {file}")
