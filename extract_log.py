

with open("/home/michele/Desktop/Progetti/RAS/empower_adf/experiments/T2_task_2/20250703-100642/logs.txt", "r", encoding="utf-8") as file:
    content = file.read()
results = analyze_log(content)

for i, group in enumerate(results):
    print(f"--- GROUP {i+1} ---")
    print(f"INPUT:\n{group['HUMAN_question']}\n")
    print(f"REASONING:\n{group['reasoning_step']}\n")
    print(f"LLM QUESTION:\n{group['LLM_question']}\n")
    print("-" * 40 + "\n")