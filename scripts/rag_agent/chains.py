from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from scripts.llm.llm_setup import get_groq_llm



def rag_chain():
    llm = get_groq_llm()
    template = """Answer the question based on the following context and the Chathistory. Especially take the latest question into consideration:

    Chathistory: {history}

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm
    return rag_chain


def off_topic_chain():
    
    off_topic_llm = get_groq_llm()

    template = """
    When a user asks an off-topic question (not related to finance):

    Provide a short, polite answer to acknowledge their query.

    Immediately and firmly remind the user that you are built specifically for finance-related queries.

    Encourage them to return to finance-related questions, since that’s where you can provide real value.

    If relevant, use the chat history to make your reminder more natural and connected to past discussions.

    Always maintain a friendly but professional tone.

    chat_history: {history}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    off_topic_rag_chain = prompt | off_topic_llm
    return off_topic_rag_chain

def internet_helper_chain():
    answer_from_internet_research_llm = get_groq_llm()

    template = """You are helpful Assistant. Help the user to get the Answer for the asked question based on the following context which is coming from internet search results.

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    internet_chain = prompt | answer_from_internet_research_llm

    return internet_chain