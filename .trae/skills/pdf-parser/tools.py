from langchain_core.tools import tool
from pypdf import PdfReader
import os

@tool
def extract_text_from_pdf(file_path: str) -> str:
    """
    【PDF Parser Skill 工具】
    用于从指定的 PDF 文件中提取纯文本内容。
    参数:
        file_path: PDF 文件的绝对或相对路径。
    返回:
        提取出的纯文本字符串，如果失败则返回错误信息。
    """
    print(f"\n  [动态 Skill 执行中...] 正在解析 PDF 文件: {file_path}")
    if not os.path.exists(file_path):
        return f"Error: 找不到文件 {file_path}。"
    
    try:
        reader = PdfReader(file_path)
        text = ""
        # 为了演示，只提取前 3 页以避免 token 过长
        max_pages = min(3, len(reader.pages))
        for i in range(max_pages):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n"
        
        return f"成功提取前 {max_pages} 页文本。内容如下：\n{text.strip()}"
    except Exception as e:
        return f"Error 解析 PDF 失败: {str(e)}"
