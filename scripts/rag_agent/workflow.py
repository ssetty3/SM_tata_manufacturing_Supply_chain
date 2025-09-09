from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
import sqlite3
from langgraph.graph import StateGraph, END

## import all the nodes
from scripts.rag_agent.nodes import (question_rewriter, question_classifier, off_topic_response, retrieve, retrieval_grader, generate_answer, 
                                     refine_question, search_internet, AgentState, on_topic_router, proceed_router)




sqlite_conn = sqlite3.connect(r"C:\Users\smm931389\Desktop\RAG_patterns\scripts\rag_agent\financebot.sqlite", check_same_thread=False)

checkpointer = SqliteSaver(sqlite_conn)



def get_compiled_workflow_graph():

    # Workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("off_topic_response", off_topic_response)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("retrieval_grader", retrieval_grader)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("refine_question", refine_question)
    #workflow.add_node("cannot_answer", cannot_answer)
    workflow.add_node("search_internet", search_internet)

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
    #workflow.add_edge("cannot_answer", "search_internet")
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

    #display_workflow_graph(get_compiled_workflow_graph())
    #roles = ["analyst", "scientist", "financial"]
    
    # To run the workflow uncomment below


    graph = get_compiled_workflow_graph()
    #input_data = {"question": HumanMessage(content="What does the company Apple do?")}
    #graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})


    input_data = {
    "question": HumanMessage(
        content="hi there My name is Sachin, how are you doing today?"
    )
    }
    graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})