import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=cW9shEB8h5E"

from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

#model.invoke("What MLB team won the World Series during the COVID-19 pandemic?")

parser = StrOutputParser()

#chain = model | parser

#print(chain.invoke("How does Link defeat Ganon in Tears of the Kingdom?"))

from langchain.prompts import ChatPromptTemplate

template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
prompt.format(context="Mary's sister is Susana", question="Who is Mary's sister?")

chain = prompt | model | parser

#print(chain.invoke({
#    "context": "Link is the protagonist in Tears of the Kingdom",
#    "question": "who is the protagonist in Tears of the Kingdom?"
#}))

#translation_prompt = ChatPromptTemplate.from_template(
#    "Translate {answer} to {language}"
#)

#from operator import itemgetter

#translation_chain = (
#    {"answer": chain, "language": itemgetter("language")} | translation_prompt | model | parser
#)

#print(translation_chain.invoke(
#    {
#        "context": "Link is the protagonist in Tears of the Kingdom. There are no other protagonists in Tears of the Kingdom.",
#        "question": "How many protagonists are there in Tears of the Kingdom?",
#        "language": "Spanish"
#    }
#))

import tempfile
import whisper
from pytube import YouTube

if not os.path.exists("c:\\poornima\\aiprojects\\rag-demo\\transcription.txt"):
    youtube = YouTube(YOUTUBE_VIDEO)
    audio = youtube.streams.filter(only_audio=True).first()

    whisper_model = whisper.load_model("base")

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        #transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()
        transcription = whisper_model.transcribe(file, fp16=False)["text"]

        with open("c:\\poornima\\aiprojects\\rag-demo\\transcription.txt", "w") as file:
            file.write(transcription)

with open("c:\\poornima\\aiprojects\\rag-demo\\transcription.txt") as file:
    transcription = file.read()
    print(transcription[:100])
    try:
        print(chain.invoke({
            "context": transcription,
            "question": "What are the types of artificial intelligence?"
        }))
    except Exception as e:
        print(e)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("c:\\poornima\\aiprojects\\rag-demo\\transcription.txt")
text_documents = loader.load()
#print(text_documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
text_splitter.split_documents(text_documents)[:5]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
#text_splitter.split_documents(text_documents)[:5]

#print(text_splitter)

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

from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstore1 = DocArrayInMemorySearch.from_texts(
    [
        "Mary's sister is Susana",
        "John and Tommy are brothers",
        "Patricia likes white cars",
        "Pedro's mother is a teacher",
        "Lucia drives an Audi",
        "Mary has two siblings",
    ],
    embedding=embeddings,
)

vectorstore1.similarity_search_with_score(query="Who is Mary's sister?", k=3)

retriever1 = vectorstore1.as_retriever()
retriever1.invoke("Who is Mary's sister?")

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

setup = RunnableParallel(context=retriever1, question=RunnablePassthrough())
setup.invoke("What color is Patricia's car?")

chain = setup | prompt | model | parser
chain.invoke("What car does Lucia drive?")

vectorstore2 = DocArrayInMemorySearch.from_documents(documents, embeddings)

chain = (
    {"context": vectorstore2.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)
#print(chain.invoke("What are the applications of artificial intelligence?"))

from langchain_pinecone import PineconeVectorStore

index_name = "ragdemo"

pinecone = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=index_name
)

pinecone.similarity_search("What are the applications of artificial intelligence?")[:3]

chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

print(chain.invoke("What are the applications of artificial intelligence?"))