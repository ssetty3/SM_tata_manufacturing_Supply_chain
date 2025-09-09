from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


from scripts.llm.llm_setup import get_groq_llm
from langchain_community.utilities import SerpAPIWrapper
from scripts.rag_agent.chains import rag_chain, off_topic_chain,internet_helper_chain
from scripts.vectorstore.faiss_vectorstore import get_retriever


## import all the nodes
from scripts.rag_agent.nodes import (question_rewriter, question_classifier, off_topic_response, retrieve, retrieval_grader, generate_answer, 
                                     refine_question, search_internet, AgentState,GradeQuestion, on_topic_router, proceed_router)



rag_chain = rag_chain()
off_topic_rag_chain = off_topic_chain()
internet_chain= internet_helper_chain()



#######
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
import sqlite3
from langgraph.graph import StateGraph, END


sqlite_conn = sqlite3.connect(r"C:\Users\smm931389\Desktop\RAG_patterns\scripts\rag_agent\main_financebot.sqlite", check_same_thread=False)

checkpointer = SqliteSaver(sqlite_conn)


## This is older way of doing it. We will do it in a better way by passing roles as argument to function.

# def get_compiled_workflow_graph():

#     # Workflow
#     workflow = StateGraph(AgentState)
#     workflow.add_node("question_rewriter", question_rewriter)
#     workflow.add_node("question_classifier", question_classifier)
#     workflow.add_node("off_topic_response", off_topic_response)
#     workflow.add_node("retrieve", retrieve)
#     workflow.add_node("retrieval_grader", retrieval_grader)
#     workflow.add_node("generate_answer", generate_answer)
#     workflow.add_node("refine_question", refine_question)
#     #workflow.add_node("cannot_answer", cannot_answer)
#     workflow.add_node("search_internet", search_internet)

#     workflow.add_edge("question_rewriter", "question_classifier")
#     workflow.add_conditional_edges(
#         "question_classifier",
#         on_topic_router,
#         {
#             "retrieve": "retrieve",
#             "off_topic_response": "off_topic_response",
#         },
#     )
#     workflow.add_edge("retrieve", "retrieval_grader")
#     workflow.add_conditional_edges(
#         "retrieval_grader",
#         proceed_router,
#         {
#             "generate_answer": "generate_answer",
#             "refine_question": "refine_question",
#             "search_internet": "search_internet",
#         },
#     )
#     workflow.add_edge("refine_question", "retrieve")
#     workflow.add_edge("generate_answer", END)
#     #workflow.add_edge("cannot_answer", "search_internet")
#     workflow.add_edge("search_internet", END)
#     workflow.add_edge("off_topic_response", END)
#     workflow.set_entry_point("question_rewriter")
#     graph = workflow.compile(checkpointer=checkpointer)

#     return graph

## Newer way of doing it by passing roles as argument to function.
def get_compiled_workflow_graph(roles: List[str]):
    # Create retriever based on user role
    retriever = get_retriever(roles=roles)

    # Workflow
    workflow = StateGraph(AgentState)

    # --- define nodes ---
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("off_topic_response", off_topic_response)

    # For retrieve we need to close over retriever
    def retrieve_node(state: AgentState):
        print("Entering retrieve")
        documents = retriever.invoke(state["rephrased_question"])
        print(f"retrieve: Retrieved {len(documents)} documents")
        state["documents"] = documents
        return state

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("retrieval_grader", retrieval_grader)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("search_internet", search_internet)

    # --- edges ---
    workflow.add_edge("question_rewriter", "question_classifier")
    workflow.add_conditional_edges(
        "question_classifier",
        on_topic_router,
        {
            "retrieve": "retrieve",
            "off_topic_response": "off_topic_response",
        },
    )
    workflow.add_edge("retrieve", "retrieval_grader")
    workflow.add_conditional_edges(
        "retrieval_grader",
        proceed_router,
        {
            "generate_answer": "generate_answer",
            "refine_question": "refine_question",
            "search_internet": "search_internet",
        },
    )
    workflow.add_edge("refine_question", "retrieve")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("search_internet", END)
    workflow.add_edge("off_topic_response", END)

    workflow.set_entry_point("question_rewriter")
    graph = workflow.compile(checkpointer=checkpointer)

    return graph


def display_workflow_graph(graph: StateGraph):
    # Visualize the graph
    from IPython.display import Image, display
    from langchain_core.runnables.graph import MermaidDrawMethod

    display(
        Image(
            graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )


if __name__ == "__main__":


    while True:
        #display_workflow_graph(get_compiled_workflow_graph())

        user_role = input("Enter role (e.g., analyst, manager, etc.): ").strip().lower()
        user_query= input("Enter your query here:") ## query :give me brekdown of all the elements development of nominal amount in the previous year ## It will fail for this query.
        graph = get_compiled_workflow_graph(roles=[user_role])
        #input_data = {"question": HumanMessage(content="What does the company Apple do?")}
        #graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})


        # input_data = {
        # "question": HumanMessage(
        #     content="hi there My name is Sachin, how are you doing today?"
        # )
        # }
        # graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})

        input_data = {
        "question": HumanMessage(
            content=user_query
        )
        }
        graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})



