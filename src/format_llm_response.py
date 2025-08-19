from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

console = Console()

def pretty_print_result(answer: str, metadata: list = None):
    """
    Pretty print the LLM's result with nice formatting.
    
    Parameters:
    -----------
    answer : str
        The text output from the LLM (can include markdown / tables).
    metadata : list of dicts
        Metadata about retrieved docs. Example:
        [
            {"file": "DB_Annual_2023.pdf", "role": "analyst"},
            {"file": "🌐 WebResult-1", "role": "web"},
            {"file": "🌐 WebResult-2", "role": "error"}
        ]
    """

    console.rule("[bold blue]💡 LLM Answer[/bold blue]")

    # If output contains markdown (tables, bold, etc.), render it
    if "|" in answer and "---" in answer:
        console.print(Markdown(answer))
    else:
        console.print(answer, style="green")

    # Print retrieved document metadata in a separate table
    if metadata:
        console.rule("[bold green]📄 Retrieved Docs Metadata[/bold green]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Role", style="yellow")

        for m in metadata:
            source = m.get("file", "N/A")
            role = m.get("role", "N/A")

            # Highlight based on role
            if role.lower() == "web":
                table.add_row(f"[yellow]{source}[/yellow]", f"[yellow]{role}[/yellow]")
            elif "error" in role.lower():
                table.add_row(f"[red]{source}[/red]", f"[red]{role}[/red]")
            else:
                table.add_row(source, role)

        console.print(table)


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
