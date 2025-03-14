from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
llm = LlamaCpp(
    model_path="model/Llama-2-7B-Chat-GGUF",
    n_gpu_layers=40,
    n_batch=512,
    verbose=False
)

#llm = Ollama(model="llama2")
# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables =["question"])
# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(llm=llm, prompt=prompt)
print("Chat bot initialized, ready to chat...")
while True:
    question=input(">")
    answer = llm_chain.run(question)
    print(answer, '\n')
