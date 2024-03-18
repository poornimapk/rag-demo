from langchain_openai.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
embedded_query = embeddings.embed_query("Who is Mary's sister?")

print(f"Embedding length: {len(embedded_query)}")
print(embedded_query[:10])

sentence1 = embeddings.embed_query("Mary's sister is Susana")
sentence2 = embeddings.embed_query("Pedro's mother is a teacher")

from sklearn.metrics.pairwise import cosine_similarity

query_sentence1_similarity = cosine_similarity([embedded_query], [sentence1])[0][0]
query_sentence2_similarity = cosine_similarity([embedded_query], [sentence2])[0][0]

print(query_sentence1_similarity, query_sentence2_similarity)