from langchain_community.llms.huggingface_hub import HuggingFaceHub
from settings import get_settings

env = get_settings()


def get_vanillaMISTRAL8x7b():
    llm = HuggingFaceHub(
        repo_id = env.MISTRAL8x7b_REPO_ID,
        huggingfacehub_api_token = env.HF_API_TOKEN,
        task = "text-generation"
    )

    return llm


prompt_template = "<s> [INST] {input_msg} [/INST] Model answer</s>"
prompt = "Hi"
input_prompt = prompt_template.format(input_msg = prompt)
out = llm.invoke(input_prompt)
print(out[len(input_prompt):])
