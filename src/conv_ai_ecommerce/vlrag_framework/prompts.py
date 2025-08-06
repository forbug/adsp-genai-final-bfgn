from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


# Define chat model
DEFAULT_LLM = ChatOllama(model="mistral", temperature=0.1)

base_system_prompt = """
    You are a product expert at Amazon and are highly knowledgeable about the available products. 
    Your job is to assist customers by answering their questions about the products.

    Use the retrieved context below to answer the user's question. Be sure to respond in a 
    way that an assistant would - be polite and accommodating. 

    Follow these important rules:

    1. Only use the information provided in the context. Do NOT make up any information.
    2. Do NOT reference the context or say “based on the context.” If needed, refer generally to “the website.”
    3. If the question is unrelated to Amazon products, politely apologize and ask the user to rephrase their question.
    4. Be concise, but specific. 
    5. If the user asks for an image, look for the image URL in the `image_url` column of the metadata. If found, display the image to the user by outputting the URL in a markdown image tag. For example: ![Product Image](image_url)

    ---  
    Context:  
    {context}

    ---  
    User question:  
    {query}


"""


def create_response_chain(prompt: str = base_system_prompt, llm=DEFAULT_LLM, output_parser=StrOutputParser):
    """
    Create a response chain for question-answering tasks.

    Args:
        prompt (str): The prompt to use for the response chain.
        llm: The language model to use for generating responses.
        output_parser: Optional; a parser to format the output.

    Returns:
        A response chain configured with the provided prompt and model.
    """

    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", "{query}"),
        ]
    )

    response_chain = (    { 
            "query": lambda x: x["query"],
            "context": lambda x: x["context"],
        }
        | response_prompt
        | llm
        | output_parser()
    )

    return response_chain
