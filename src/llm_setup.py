from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def get_groq_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",  # or another Groq model you have access to
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=1024,
    )

# === Test LLM with "hello" ===
if __name__ == "__main__":
    llm = get_groq_llm()
    message = HumanMessage(content="hello")

    # 👇 Use invoke() for future-proof, recommended usage
    response = llm.invoke([message])

    print("🔁 Response from Groq LLM:")
    print(response.content)