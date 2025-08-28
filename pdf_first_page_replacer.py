#!/usr/bin/env python3
"""
PDF First Page Replacer
Um CLI interativo para substituir a primeira página de múltiplos PDFs
"""

import os
import sys
import shlex
from pathlib import Path
from typing import List, Optional

import typer
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.panel import Panel
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter

app = typer.Typer(
    name="pdf-replacer",
    help="🔄 Substitui a primeira página de múltiplos PDFs por uma página de referência",
    add_completion=False,
    rich_markup_mode="rich"
)

console = Console()


def normalize_path_input(path_str: str) -> str:
    """Normaliza entradas de caminho removendo aspas, decodificando escapes de shell e expandindo ~"""
    s = path_str.strip()
    # Remove aspas ao redor
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    # Se houver barras invertidas (escapes de shell), tenta decodificar com shlex
    if "\\" in s:
        try:
            parts = shlex.split(s)
            if len(parts) == 1:
                s = parts[0]
        except Exception:
            # Se falhar a decodificação, mantém o valor original
            pass
    # Expande ~ para o diretório do usuário
    s = os.path.expanduser(s)
    return s

def validate_pdf_file(file_path: str) -> bool:
    """Valida se um arquivo é um PDF válido"""
    try:
        with open(file_path, 'rb') as file:
            PdfReader(file)
        return True
    except Exception:
        return False

def find_pdf_files(directory: Path) -> List[Path]:
    """Encontra todos os arquivos PDF em um diretório"""
    pdf_files = []
    for file_path in directory.rglob("*.pdf"):
        if file_path.is_file() and validate_pdf_file(str(file_path)):
            pdf_files.append(file_path)
    return sorted(pdf_files)

def replace_first_page(source_pdf: Path, replacement_pdf: Path, output_pdf: Path) -> bool:
    """Substitui a primeira página de um PDF"""
    try:
        with open(replacement_pdf, 'rb') as replacement_file, \
             open(source_pdf, 'rb') as source_file:

            replacement_reader = PdfReader(replacement_file)
            replacement_page = replacement_reader.pages[0]

            source_reader = PdfReader(source_file)
            
            # Cria um novo PDF writer
            writer = PdfWriter()
            
            # Adiciona a página de substituição primeiro
            writer.add_page(replacement_page)
            
            # Adiciona as páginas restantes (pulando a primeira)
            for i in range(1, len(source_reader.pages)):
                writer.add_page(source_reader.pages[i])
            
            # Salva o novo PDF
            with open(output_pdf, 'wb') as output_file:
                writer.write(output_file)
        
        return True
    except Exception as e:
        console.print(f"❌ Erro ao processar {source_pdf.name}: {str(e)}", style="red")
        return False

@app.command()
def replace(
    pdf_directory: Optional[str] = typer.Argument(None, help="Caminho para a pasta com os PDFs"),
    replacement_pdf: Optional[str] = typer.Argument(None, help="Caminho para o PDF de substituição"),
    output_directory: Optional[str] = typer.Option(None, "--output", "-o", help="Pasta de saída (padrão: mesma pasta dos originais)"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Criar backup dos arquivos originais"),
    interactive: bool = typer.Option(True, "--interactive/--batch", help="Modo interativo")
):
    """
    🔄 Substitui a primeira página de todos os PDFs em uma pasta
    """
    
    # Banner inicial
    console.print(Panel.fit(
        "[bold blue]PDF First Page Replacer[/bold blue]\n"
        "[dim]Substitui a primeira página de múltiplos PDFs[/dim]",
        border_style="blue"
    ))
    
    # Solicita o diretório dos PDFs se não fornecido
    if not pdf_directory:
        pdf_directory = Prompt.ask(
            "\n📁 Digite o caminho para a pasta com os PDFs",
            default="."
        )
    
    pdf_directory = normalize_path_input(pdf_directory)
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        console.print(f"❌ Diretório não encontrado: {pdf_directory}", style="red")
        raise typer.Exit(1)
    
    # Solicita o PDF de substituição se não fornecido
    if not replacement_pdf:
        replacement_pdf = Prompt.ask("\n📄 Digite o caminho para o PDF de substituição (1 página)")
    
    replacement_pdf = normalize_path_input(replacement_pdf)
    replacement_path = Path(replacement_pdf)
    if not replacement_path.exists() or not validate_pdf_file(str(replacement_path)):
        console.print(f"❌ PDF de substituição inválido: {replacement_pdf}", style="red")
        raise typer.Exit(1)
    
    # Verifica se o PDF de substituição tem apenas 1 página
    try:
        with open(replacement_path, 'rb') as file:
            reader = PdfReader(file)
            if len(reader.pages) != 1:
                console.print(f"⚠️ Aviso: O PDF de substituição tem {len(reader.pages)} páginas. Apenas a primeira será usada.", style="yellow")
    except Exception as e:
        console.print(f"❌ Erro ao verificar PDF de substituição: {e}", style="red")
        raise typer.Exit(1)
    
    # Encontra todos os PDFs
    console.print("\n🔍 Procurando arquivos PDF...")
    pdf_files = find_pdf_files(pdf_dir)
    
    if not pdf_files:
        console.print("❌ Nenhum arquivo PDF válido encontrado", style="red")
        raise typer.Exit(1)
    
    # Mostra tabela com os PDFs encontrados
    table = Table(title=f"📋 {len(pdf_files)} PDF(s) encontrado(s)")
    table.add_column("Arquivo", style="cyan")
    table.add_column("Páginas", justify="center")
    table.add_column("Tamanho", justify="right")
    
    for pdf_file in pdf_files[:10]:  # Mostra apenas os primeiros 10
        try:
            with open(pdf_file, 'rb') as file:
                reader = PdfReader(file)
                pages = len(reader.pages)
                size = f"{pdf_file.stat().st_size / 1024:.1f} KB"
                table.add_row(pdf_file.name, str(pages), size)
        except:
            table.add_row(pdf_file.name, "?", "?")
    
    if len(pdf_files) > 10:
        table.add_row("...", "...", "...")
        table.add_row(f"[dim]+ {len(pdf_files) - 10} arquivo(s)[/dim]", "", "")
    
    console.print(table)
    
    # Confirmação interativa
    if interactive:
        if not Confirm.ask(f"\n❓ Deseja substituir a primeira página de {len(pdf_files)} PDF(s)?"):
            console.print("❌ Operação cancelada", style="yellow")
            raise typer.Exit(0)
    
    # Define diretório de saída
    if output_directory:
        output_directory = normalize_path_input(output_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = pdf_dir
    
    # Processa os arquivos
    console.print(f"\n🚀 Processando {len(pdf_files)} arquivo(s)...")
    
    success_count = 0
    error_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        
        task = progress.add_task("Processando PDFs...", total=len(pdf_files))
        
        for pdf_file in pdf_files:
            progress.update(task, description=f"Processando {pdf_file.name}...")
            
            # Define arquivo de saída
            if output_dir == pdf_dir:
                if backup:
                    # Cria backup
                    backup_file = pdf_file.parent / f"{pdf_file.stem}_backup{pdf_file.suffix}"
                    pdf_file.replace(backup_file)
                    output_file = pdf_file
                else:
                    output_file = pdf_file.parent / f"{pdf_file.stem}_modified{pdf_file.suffix}"
            else:
                output_file = output_dir / pdf_file.name
            
            # Substitui a primeira página
            if replace_first_page(pdf_file if not backup else backup_file, replacement_path, output_file):
                success_count += 1
                console.print(f"✅ {pdf_file.name}", style="green")
            else:
                error_count += 1
            
            progress.update(task, advance=1)
    
    # Relatório final
    console.print(f"\n📊 [bold]Relatório Final[/bold]")
    console.print(f"✅ Sucessos: {success_count}", style="green")
    if error_count > 0:
        console.print(f"❌ Erros: {error_count}", style="red")
    
    if backup and success_count > 0:
        console.print(f"💾 Backups salvos com sufixo '_backup'", style="blue")
    
    console.print(f"📁 Arquivos salvos em: {output_dir.absolute()}", style="cyan")
    
    if success_count > 0:
        console.print("\n🎉 [bold green]Operação concluída com sucesso![/bold green]")
    else:
        console.print("\n❌ [bold red]Nenhum arquivo foi processado com sucesso[/bold red]")
        raise typer.Exit(1)

@app.command()
def info():
    """ℹ️ Mostra informações sobre o programa"""
    console.print(Panel(
        "[bold blue]PDF First Page Replacer v1.0[/bold blue]\n\n"
        "[bold]Funcionalidades:[/bold]\n"
        "• Substitui a primeira página de múltiplos PDFs\n"
        "• Suporte a backup automático\n"
        "• Interface interativa e amigável\n"
        "• Validação de arquivos PDF\n"
        "• Relatórios detalhados\n\n"
        "[bold]Dependências:[/bold]\n"
        "• typer\n"
        "• rich\n"
        "• PyPDF2\n\n"
        "[bold]Exemplos de uso:[/bold]\n"
        "[cyan]python pdf_replacer.py replace[/cyan]\n"
        "[cyan]python pdf_replacer.py replace ./pdfs nova_primeira_pagina.pdf[/cyan]\n"
        "[cyan]python pdf_replacer.py replace --output ./output --no-backup[/cyan]",
        border_style="blue"
    ))

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n❌ Operação interrompida pelo usuário", style="yellow")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ Erro inesperado: {e}", style="red")
        sys.exit(1)