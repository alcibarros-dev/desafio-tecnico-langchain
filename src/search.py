from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import SetupError, get_llm, get_vector_store

TOP_K = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt(question=None):
    """Monta a chain de RAG. Com `question`, já devolve a resposta; sem, devolve a chain."""
    try:
        store = get_vector_store()
        llm = get_llm()
    except SetupError as e:
        print(f"Erro: {e}")
        return None

    def retrieve_context(pergunta):
        results = store.similarity_search_with_score(pergunta, k=TOP_K)
        return "\n\n".join(doc.page_content for doc, _score in results)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"contexto": RunnableLambda(retrieve_context), "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if question is not None:
        return chain.invoke(question)
    return chain
