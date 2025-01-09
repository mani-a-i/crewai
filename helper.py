def mistral_prompt_template(prompt):
    
    prompt_template = "<s> [INST] {input_prompt} [/INST] Model answer</s>"
    return prompt_template.format(input_prompt=prompt)