def mistral_prompt_template(prompt):
    
    prompt_template = "<s> [INST] {input_prompt} [/INST] Model answer</s>"
    return prompt_template.format(input_prompt=prompt)

def zephyr_prompt_template(prompt):
    prompt_template = "</s><|user|> {input_prompt} </s><|assistant|>"
    return prompt_template.format(input_prompt = prompt) 

def llama_prompt_template(prompt):
    prompt_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|> {input_prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt_template.format(input_prompt = prompt) 