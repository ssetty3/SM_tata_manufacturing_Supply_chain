from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.llm_setup import get_groq_llm

# === Summarization Prompt ===
summary_prompt = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template="""
You are a memory manager. Update the running summary of the conversation.

Previous summary:
{summary}

New lines to add:
{new_lines}

Updated summary:
"""
)

class ConversationSummary:
    def __init__(self, summarize_every=4):
        """
        Parameters
        ----------
        summarize_every : int
            Number of turns (messages) after which summarization is triggered.
            Example: 4 = summarize after 4 new turns (≈ 2 user+assistant exchanges).
        """
        self.summary = "Conversation starts here."
        self.recent_turns = []
        self.summarize_every = summarize_every

    def update_summary(self, llm):
        """Perform summarization and reset buffer"""
        formatted_prompt = summary_prompt.format(
            summary=self.summary,
            new_lines="\n".join([f"{t['role']}: {t['content']}" for t in self.recent_turns])
        )
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        new_summary = response.content.strip()

        print("🔁 Summarizing memory...")
        print(f"   ➡️ Previous summary: {self.summary}")
        print(f"   ➡️ Added turns: {[t['role'] for t in self.recent_turns]}")
        print(f"   ✅ New summary: {new_summary}\n")

        self.summary = new_summary
        self.recent_turns = []  # reset buffer

    def add(self, role, content, llm):
        """Add a new turn and check if summarization is needed"""
        self.recent_turns.append({"role": role, "content": content})
        print(f"📝 Added turn: {role} → {content[:60]}...")

        if len(self.recent_turns) >= self.summarize_every:
            print(f"⚡ Threshold reached ({self.summarize_every} turns). Triggering summarization.")
            self.update_summary(llm)

    def get_context(self):
        """Return full memory context (summary + recent turns)"""
        return self.summary + "\n" + "\n".join(
            [f"{t['role']}: {t['content']}" for t in self.recent_turns]
        )


# === Example usage ===
if __name__ == "__main__":
    llm = get_groq_llm()
    memory = ConversationSummary(summarize_every=4)  # configurable N

    # simulate chat
    memory.add("user", "Hi, can you tell me about Tesla?", llm)
    memory.add("assistant", "Tesla is an EV company founded by Elon Musk.", llm)
    memory.add("user", "What is their latest quarterly revenue?", llm)
    memory.add("assistant", "Tesla reported $25B revenue last quarter.", llm)  # ⚡ triggers summarization
    memory.add("user", "Great, what about their competitors?", llm)

    print("\n=== Current Memory Context ===")
    print(memory.get_context())
