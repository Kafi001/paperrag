# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) addresses two limitations of language
models: their knowledge is frozen at training time, and they confabulate
confidently when they don't know something. RAG grounds generation in retrieved
documents, so answers can cite sources and stay current without retraining.

A RAG system has four stages. Ingestion loads documents and splits them into
chunks. Embedding converts each chunk into a dense vector capturing semantic
meaning. Retrieval embeds the user's query and finds the nearest chunks by
vector similarity. Generation passes those chunks to a language model as
context, with instructions to answer only from that context.

Chunking is the most underrated design decision. Chunks that are too large
dilute relevance, because the retriever matches on the chunk as a whole and a
long passage may be mostly irrelevant. Chunks that are too small lose the
surrounding context needed to make sense of them. Overlapping chunks mitigate
the boundary problem, where a sentence answering the question is split across
two chunks and neither retrieves well.

Cosine similarity is the usual distance measure for normalised sentence
embeddings, since it compares direction rather than magnitude. Approximate
nearest neighbour indexes such as HNSW make search fast at scale by trading
exact results for speed.

Evaluating RAG requires separating retrieval failures from generation failures.
If the correct chunk was never retrieved, no amount of prompt engineering will
fix the answer. Inspecting retrieved context before blaming the model is the
first debugging step.
