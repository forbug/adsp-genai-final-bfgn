from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

import pandas as pd
import numpy as np


# Define chat model
DEFAULT_LLM = ChatOllama(model="gpt-oss", temperature=0.1)

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
    5. If the image URL in the `image_url` column of the metadata is populated, display the image to the user by outputting the URL in a markdown image tag. For example: ![Product Image](image_url)
    6. If no user query is provided, describe the relevant products in the context.
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



def retrieve_relevant_products(multimodal_index, text_index, image_index, query_embedding: str | None = None, image_embedding: str | None = None):
    if query_embedding is not None and image_embedding is not None:
        print("Using multi-modal collection.")
        combined_embedding = np.hstack([query_embedding[0], image_embedding[0]])
        res = multimodal_index.query(combined_embedding, n_results=3)
    elif query_embedding is not None:
        print("Using text-only collection.")
        res = text_index.query(query_embedding, n_results=3)
    elif image_embedding is not None:
        print("Using image-only collection.")
        res = image_index.query(image_embedding, n_results=3)
    else:
        raise ValueError("Query or image must be passed!")
    
    retrieved_items = res['metadatas'][0]

    # Display results
    print("Retrieved Results:")
    for row in retrieved_items:
        print(row)

    return retrieved_items




def generate_response(user_query: str, retrieved_items: pd.DataFrame):
    """Generate a response based on the retrieved documents and optional Amazon data."""

    response_chain = create_response_chain()
    
    # Build context from local results
    context_list = []
    for product in retrieved_items:
        context_str = product['search_text']
        if 'image_url' in product and pd.notna(product['image_url']):
             context_str += f"\nImage URL: {product['image_url']}"
        context_list.append(context_str)
    
    local_context = "\n\n---\n\n".join(context_list)
    
    # Combine local and Amazon context
    full_context = local_context
    
    response = response_chain.invoke({
        "query": user_query,
        "context": full_context,
    })

    return response



