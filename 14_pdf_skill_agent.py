from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import importlib
import importlib.util
import os
import sys

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

# =======================================================
# 1. 动态工具发现与加载 (Dynamic Skill Loading)
# =======================================================
def load_skills_from_directory(skills_dir: str):
    """
    扫描指定目录（如 .trae/skills/），动态加载里面定义的所有 @tool。
    这就是企业级应用中“热插拔”技能系统的基础实现。
    """
    loaded_tools = []
    
    if not os.path.exists(skills_dir):
        print(f"[系统] 未找到技能目录: {skills_dir}")
        return loaded_tools

    print(f"\n[系统] 正在从 {skills_dir} 扫描并动态加载 Skills...")
    
    # 遍历 skills 目录下的所有子文件夹
    for skill_name in os.listdir(skills_dir):
        skill_path = os.path.join(skills_dir, skill_name)
        if os.path.isdir(skill_path):
            tools_file = os.path.join(skill_path, "tools.py")
            if os.path.exists(tools_file):
                # 动态加载 Python 模块
                module_name = f"dynamic_skill_{skill_name}"
                spec = importlib.util.spec_from_file_location(module_name, tools_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    # 遍历模块中的所有属性，找到被 @tool 装饰的函数
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        # LangChain 的 @tool 装饰器会返回一个 BaseTool 的子类实例
                        if hasattr(attr, "name") and hasattr(attr, "description") and hasattr(attr, "invoke"):
                            # 简单的判断：如果它有 invoke 方法且不是模块本身，我们认为它是一个 Tool
                            if callable(getattr(attr, "invoke", None)):
                                loaded_tools.append(attr)
                                print(f"  -> 成功挂载工具: [{attr.name}] 来自技能包 '{skill_name}'")
                                
    return loaded_tools

# =======================================================
# 2. 模拟创建一个测试用的 PDF 文件
# =======================================================
def create_dummy_pdf(file_path: str):
    """使用 fpdf 快速生成一个测试用的 PDF，如果环境中没有 fpdf，则提示用户准备文件。"""
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        # 兼容中文的设置比较复杂，这里用英文演示
        pdf.set_font("Arial", size=15)
        pdf.cell(200, 10, txt="Welcome to LangChain PDF Skill Test!", ln=True, align='C')
        pdf.cell(200, 10, txt="This is a dummy PDF file.", ln=True, align='C')
        pdf.cell(200, 10, txt="The secret password is: TRAE-AI-2026", ln=True, align='C')
        pdf.output(file_path)
        print(f"[系统] 已自动生成测试 PDF 文件: {file_path}")
    except ImportError:
        print("[系统] 未安装 fpdf，跳过自动生成 PDF。请确保目录下有该文件。")

# =======================================================
# 3. 运行 Agent
# =======================================================
def main():
    print("=== 开始运行 基于 Agent 调用 PDF 解析 Skill 的演示 ===")
    
    # 准备环境和模型
    dummy_pdf_path = "sample_document.pdf"
    create_dummy_pdf(dummy_pdf_path)
    
    llm = get_model("openai")
    
    # 【动态加载核心】：不再硬编码 tools = [extract_text_from_pdf]
    # 而是让 Agent 在启动时扫描工作区下的技能目录，动态挂载工具
    skills_directory = os.path.join(os.path.dirname(__file__), ".trae", "skills")
    tools = load_skills_from_directory(skills_directory)
    
    if not tools:
        print("[警告] 没有扫描到任何可用的技能工具。Agent 可能无法完成复杂任务。")
    
    # 创建带有动态技能集的 Agent
    agent_executor = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt="你是一个善于阅读文档的智能助手。当你需要了解 PDF 内容时，必须调用提供的工具。"
    )

    # 发起提问
    question = f"请帮我读取当前目录下的 {dummy_pdf_path} 文件，并告诉我里面提到的秘密密码（secret password）是什么？"
    print(f"\nUser: {question}")
    
    # 执行并获取结果
    res = agent_executor.invoke({"messages": [HumanMessage(content=question)]})
    
    print(f"\nAI 最终回复:\n{res['messages'][-1].content}")

if __name__ == "__main__":
    main()
