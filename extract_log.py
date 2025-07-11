import re 

def analyze_log(text_content):
    """
    Analizza il testo di una conversazione sequenziale, dove la risposta
    di un group diventa l'input per il successivo.
    """
    group_QA = re.split(r'(?=LLM Response #\d+:|Initial LLM Response:)', text_content)
    
    results = []
    # Correggo il pattern per catturare le risposte umane
    human_answer = re.findall(r"Human Answer #\d+[:\s]*([^\n]*)", text_content)
    
    human_question = re.search(r"Human Answer #\d+[:\s]*([^\n]*)", text_content)
    current_input = human_question.group(1).strip() if human_question else "N/A"

    for i, group in enumerate(group_QA):
        if "LLM Response" not in group and "Initial LLM Response" not in group:
            continue

        reasoning = ""
        reason_match = re.search(r"REASON phase:(.*?)(?=THINK phase:)", group, re.DOTALL)
        if reason_match:
            reasoning += "REASON:\n" + reason_match.group(1).strip() + "\n\n"
            
        think_match = re.search(r"THINK phase:(.*?)(?=ASK phase:)", group, re.DOTALL)
        if think_match:
            reasoning += "THINK:\n" + think_match.group(1).strip() + "\n\n"

        ask_match = re.search(r"ASK phase:(.*?)(?=<PLAN>|\n\nIf the answer confirms)", group, re.DOTALL)
        if ask_match:
            reasoning += "ASK:\n" + ask_match.group(1).strip()
        
        question_LLM = re.search(r"<QUESTION>(.*?)</QUESTION>", group, re.DOTALL)
        domanda_generata = question_LLM.group(1).strip() if question_LLM else "Nothing else to ask. Generate the plan."

        results.append({
            "HUMAN_question": current_input,
            "reasoning_step": reasoning.strip(),
            "LLM_question": domanda_generata
        })

        if i < len(human_answer):
            current_input = human_answer[i]

    return results

with open("/home/michele/Desktop/Progetti/RAS/empower_adf/experiments/T3_task_3/20250704-115053/logs.txt", "r", encoding="utf-8") as file:
    content = file.read()
results = analyze_log(content)
print(results[len(results)-1])
