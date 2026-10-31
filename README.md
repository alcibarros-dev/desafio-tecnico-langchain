# Ingestão e Busca Semântica com LangChain e Postgres

Solução do desafio do módulo **Introdução ao LangChain** (Full Cycle).

O projeto faz duas coisas:

- **Ingestão:** lê o `document.pdf`, divide o texto em chunks de 1000 caracteres com overlap de 150, gera um embedding por chunk e salva os vetores no PostgreSQL com pgVector.
- **Busca:** um chat no terminal que vetoriza a pergunta, busca os 10 chunks mais relevantes (`similarity_search_with_score(query, k=10)`), monta o prompt e chama a LLM. A resposta é baseada apenas no conteúdo do PDF.

Exemplo:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Tecnologias

- Python 3.13
- LangChain (`langchain-openai`, `langchain-google-genai`, `langchain-postgres`, `langchain-community`)
- PostgreSQL 17 + pgVector (via Docker Compose)
- OpenAI **ou** Google Gemini para embeddings e LLM

## Estrutura

```
├── docker-compose.yml    # PostgreSQL + pgVector
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── config.py         # Leitura do .env, escolha do provedor, embeddings, LLM e PGVector
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Prompt e chain de busca (RAG)
│   └── chat.py           # CLI para interação com o usuário
├── document.pdf          # PDF para ingestão
└── README.md
```

## Modelos

O desafio não fixa modelos. Os padrões abaixo foram escolhidos consultando a documentação oficial de cada provedor em outubro de 2026, priorizando os modelos mais leves e baratos:

| Provedor | Embeddings | LLM |
|----------|-----------|-----|
| OpenAI | `text-embedding-3-small` (1536 dimensões) | `gpt-6-luna` |
| Gemini | `models/gemini-embedding-001` (3072 dimensões) | `gemini-3.5-flash-lite` |

No Gemini foi usado o `gemini-embedding-001` e não o `gemini-embedding-2`: o `-2` não aceita o parâmetro `task_type`, que o `GoogleGenerativeAIEmbeddings` envia para diferenciar documento e pergunta.

Todos os modelos podem ser trocados pelo `.env` (veja abaixo).

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Uma API Key da [OpenAI](https://platform.openai.com/api-keys) ou do [Google AI Studio](https://aistudio.google.com/apikey)

## Como executar

### 1. Clonar o repositório e criar o ambiente virtual

```bash
git clone <url-deste-repositorio>
cd <pasta-do-repositorio>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar as variáveis de ambiente

```bash
cp .env.example .env
```

Abra o `.env` e preencha **uma** das chaves (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`). O restante já vem com valores que funcionam com o `docker-compose.yml`.

| Variável | Descrição |
|----------|-----------|
| `LLM_PROVIDER` | `openai` ou `gemini`. Se ficar vazio, usa o provedor cuja chave estiver preenchida (OpenAI tem prioridade). |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | Chave do provedor escolhido. |
| `OPENAI_EMBEDDING_MODEL` / `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings. |
| `OPENAI_LLM_MODEL` / `GOOGLE_LLM_MODEL` | Modelo usado para responder. |
| `DATABASE_URL` | Conexão com o Postgres (`postgresql+psycopg://postgres:postgres@localhost:5432/rag`). |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection onde os chunks são salvos. |
| `PDF_PATH` | Caminho do PDF (relativo à raiz do projeto ou absoluto). Padrão: `document.pdf`. |

### 3. Subir o banco de dados

```bash
docker compose up -d
```

O serviço `bootstrap_vector_ext` cria a extensão `vector` e encerra; o container `postgres_rag` fica rodando na porta 5432.

### 4. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada:

```
Carregando PDF: .../document.pdf
34 páginas divididas em 67 chunks.
Ingestão concluída: 67 chunks salvos no PostgreSQL.
```

A ingestão recria a collection a cada execução, então rodar o script de novo não duplica chunks.

### 5. Rodar o chat

```bash
python src/chat.py
```

Digite as perguntas após `PERGUNTA:`. Para sair, digite `sair` ou pressione `Ctrl+C` / `Ctrl+D`.

## Trocando o modelo de embeddings ou o provedor

Modelos diferentes geram vetores com dimensões diferentes, e a busca só funciona se a pergunta e os chunks forem vetorizados pelo mesmo modelo. Depois de trocar o provedor ou o modelo de embeddings no `.env`, rode a ingestão de novo:

```bash
python src/ingest.py
```

Como a ingestão apaga e recria a collection, os vetores antigos são descartados. Se quiser zerar o banco por completo:

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```

## Como funciona

1. `ingest.py` carrega o PDF com `PyPDFLoader`, divide com `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)` e salva com `PGVector.add_documents`, que gera os embeddings.
2. `search.py` monta uma chain LCEL: a pergunta é usada em `similarity_search_with_score(pergunta, k=10)`, os 10 chunks são concatenados no `{contexto}` do prompt do desafio, e o resultado vai para a LLM.
3. `chat.py` chama `search_prompt()` para obter a chain e fica em loop lendo perguntas e imprimindo as respostas.

Se faltar alguma variável no `.env` ou o banco não estiver acessível, os scripts mostram uma mensagem explicando o problema em vez de um traceback.
