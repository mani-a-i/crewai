def mistral_prompt_template(prompt:str) -> str:
    
    prompt_template:str = "[INST] {input_prompt} [/INST]"
    return prompt_template.format(input_prompt=prompt)

def zephyr_prompt_template(prompt:str) -> str:
    prompt_template:str = "</s><|user|> {input_prompt} </s><|assistant|>"
    return prompt_template.format(input_prompt = prompt) 

def llama_prompt_template(prompt:str) -> str:
    prompt_template:str = "<|begin_of_text|><|start_header_id|>user<|end_header_id|> {input_prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt_template.format(input_prompt = prompt) 