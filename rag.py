import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=cdiD-9MMpb0"

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
prompt.format(context="Link is the protagonist in Tears of the Kingdom", question="who is the protagonist in Tears of the Kingdom?")

chain = prompt | model | parser

print(chain.invoke({
    "context": "Link is the protagonist in Tears of the Kingdom",
    "question": "who is the protagonist in Tears of the Kingdom?"
}))

translation_prompt = ChatPromptTemplate.from_template(
    "Translate {answer} to {language}"
)

from operator import itemgetter

translation_chain = (
    {"answer": chain, "language": itemgetter("language")} | translation_prompt | model | parser
)

print(translation_chain.invoke(
    {
        "context": "Link is the protagonist in Tears of the Kingdom. There are no other protagonists in Tears of the Kingdom.",
        "question": "How many protagonists are there in Tears of the Kingdom?",
        "language": "Spanish"
    }
))