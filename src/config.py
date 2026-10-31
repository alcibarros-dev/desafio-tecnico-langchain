import os

from dotenv import load_dotenv
from langchain_postgres import PGVector
from sqlalchemy.exc import OperationalError

load_dotenv()

DEFAULT_MODELS = {
    "openai": {
        "embedding": "text-embedding-3-small",
        "llm": "gpt-6-luna",
    },
    "gemini": {
        "embedding": "models/gemini-embedding-001",
        "llm": "gemini-3.5-flash-lite",
    },
}

API_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


class SetupError(Exception):
    """Erro de configuração (.env) ou de conexão com o banco."""


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise SetupError(f"A variável {name} não está definida no arquivo .env.")
    return value


def get_provider():
    """Usa LLM_PROVIDER se definido; senão, escolhe pelo que tiver chave configurada."""
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()

    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            raise SetupError("Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env.")

    if provider not in DEFAULT_MODELS:
        raise SetupError(f"LLM_PROVIDER inválido: '{provider}'. Use 'openai' ou 'gemini'.")

    require_env(API_KEY_VARS[provider])
    return provider


def get_embeddings():
    provider = get_provider()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_MODELS["openai"]["embedding"]
        return OpenAIEmbeddings(model=model)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = os.getenv("GOOGLE_EMBEDDING_MODEL") or DEFAULT_MODELS["gemini"]["embedding"]
    return GoogleGenerativeAIEmbeddings(model=model)


def get_llm():
    provider = get_provider()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_LLM_MODEL") or DEFAULT_MODELS["openai"]["llm"]
        return ChatOpenAI(model=model)

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = os.getenv("GOOGLE_LLM_MODEL") or DEFAULT_MODELS["gemini"]["llm"]
    return ChatGoogleGenerativeAI(model=model, temperature=0)


def get_vector_store(embeddings=None, pre_delete_collection=False):
    embeddings = embeddings or get_embeddings()
    collection_name = require_env("PG_VECTOR_COLLECTION_NAME")
    connection = require_env("DATABASE_URL")

    try:
        return PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
            pre_delete_collection=pre_delete_collection,
        )
    except Exception as e:
        # O PGVector embrulha a falha de conexão numa Exception genérica.
        if isinstance(e, OperationalError) or isinstance(e.__cause__, OperationalError):
            raise SetupError(
                "Não foi possível conectar ao banco de dados. "
                "Verifique se o container está rodando (docker compose up -d) e o DATABASE_URL."
            ) from e
        raise
