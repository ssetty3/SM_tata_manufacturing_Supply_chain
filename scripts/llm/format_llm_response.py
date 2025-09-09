from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
from typing import List, Dict, Any

console = Console()

def pretty_print_result(
    answer: str,
    metadata: List[Dict[str, Any]] = None,
    traces: List[Dict[str, Any]] = None
):
    """
    Pretty print the LLM's result with nice formatting.

    Parameters
    ----------
    answer : str
        The text output from the LLM (can include markdown / tables).
    metadata : list of dicts, optional
        Metadata about retrieved docs.
        Example:
            [
                {"file": "DB_Annual_2023.pdf", "role": "analyst"},
                {"file": "🌐 WebResult-1", "role": "web"},
                {"file": "🌐 WebResult-2", "role": "error"}
            ]
    traces : list of dicts, optional
        Workflow trace logs.
        Example:
            [
                {"step": "cache_check", "details": {"hit": False}},
                {"step": "retrieve", "details": {"num_docs": 3}}
            ]
    """

    # --- Answer ---
    console.rule("[bold blue]💡 LLM Answer[/bold blue]")

    if "|" in answer and "---" in answer:  # looks like Markdown table
        console.print(Markdown(answer))
    else:
        console.print(answer, style="green")

    # --- Metadata ---
    if metadata:
        console.rule("[bold green]📄 Retrieved Docs Metadata[/bold green]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Role", style="yellow")

        for m in metadata:
            source = m.get("file", "N/A")
            role = m.get("role", "N/A")

            if role.lower() == "web":
                table.add_row(f"[yellow]{source}[/yellow]", f"[yellow]{role}[/yellow]")
            elif "error" in role.lower():
                table.add_row(f"[red]{source}[/red]", f"[red]{role}[/red]")
            else:
                table.add_row(source, role)

        console.print(table)

    # --- Traces ---
    if traces:
        console.rule("[bold purple]🧭 Execution Trace[/bold purple]")

        trace_table = Table(show_header=True, header_style="bold magenta")
        trace_table.add_column("Step", style="cyan", no_wrap=True)
        trace_table.add_column("Details", style="white")

        for t in traces:
            step = t.get("step", "N/A")
            details = t.get("details", {})

            # Format details dictionary into nice key=value pairs
            if isinstance(details, dict):
                detail_str = "\n".join([f"[green]{k}[/green]: {v}" for k, v in details.items()])
            else:
                detail_str = str(details)

            trace_table.add_row(step, detail_str)

        console.print(trace_table)


# === Example usage ===
if __name__ == "__main__":
    llm_answer = """**Key financial risks highlighted**  

| Source | Core financial risk(s) | Why it matters |
|--------|------------------------|----------------|
| Deutsche Bank | Credit Risk, Liquidity, IT Risk | Impacts profitability, compliance |
| Unilever | Treasury Risk, Ethical Risk | Impacts cost of capital, reputation |  

**Bottom line:** Risks directly affect profitability and capital adequacy.
"""

    metadata = [
        {"file": "DB_Annual_2023.pdf", "role": "analyst"},
        {"file": "Unilever_Annual_2024.pdf", "role": "financial"},
        {"file": "🌐 WebResult-1", "role": "web"},
        {"file": "🌐 WebResult-2", "role": "error"},
    ]

    pretty_print_result(llm_answer, metadata)
