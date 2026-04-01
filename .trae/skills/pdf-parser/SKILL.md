---
name: "pdf-parser"
description: "Parses and extracts text content from PDF files. Invoke this skill when the user asks to read, analyze, or extract text from a PDF document."
---

# PDF Parser Skill

This skill provides the ability to extract text from PDF files. It uses `PyPDF2` (or a similar library) under the hood to read the content of a given PDF file and return it as a string for further analysis.

## Usage Guidelines

1. **When to use**: Whenever the user provides a PDF file path and wants to know its contents, summarize it, or search for specific information inside it.
2. **Requirements**: The target project must have a PDF parsing library installed (e.g., `pip install PyPDF2` or `pip install pypdf`).

## Example Python Implementation

To use this skill programmatically, you can define a tool like this:

```python
from langchain_core.tools import tool
from pypdf import PdfReader
import os

@tool
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts all text content from a specified PDF file.
    Args:
        file_path: The absolute or relative path to the PDF file.
    """
    if not os.path.exists(file_path):
        return f"Error: The file {file_path} does not exist."
    
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"
```
