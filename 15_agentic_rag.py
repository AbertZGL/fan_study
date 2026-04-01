import importlib
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

# 对应课程：4.1 RAG基础概述 与 4.2 AgenticRAG概念介绍与自定义RAG检索工具Tool

# ======================================================================
# 什么是 Agentic RAG？
# 传统的 RAG（Retrieval-Augmented Generation）是“被动”的：
#   用户提问 -> 检索向量数据库 -> 把查到的所有内容塞给大模型 -> 大模型生成回答。
# 这种方式很死板：如果用户问“你好”，传统 RAG 也会傻乎乎地去库里搜“你好”；如果搜出来的信息不够，它也无法再搜一次。
#
# Agentic RAG 则是“主动”的：
#   我们把检索器（Retriever）包装成一个 Tool（工具），交给 Agent。
#   Agent 会自主思考：
#   1. 我需要检索吗？（问“你好”时不检索，问业务问题时检索）
#   2. 我该用什么关键词检索？（把用户复杂的问题拆解提炼成高质检索词）
#   3. 搜出来的信息够了吗？（如果不够，Agent 可以换个关键词再搜一次，甚至调用计算器算一下）
# ======================================================================

# 模拟一个底层的 RAG 检索器（真实情况中它是 FAISS、Milvus 等）
def mock_vector_retriever(query: str) -> str:
    if "年假" in query:
        return "公司规定入职满一年可享受 5 天带薪年假。"
    elif "报销" in query:
        return "晚上 10 点后打车可以全额报销。"
    else:
        return "知识库中未找到相关内容。"

# 对应课程：4.2 自定义RAG检索工具Tool
@tool
def company_policy_search(query: str) -> str:
    """
    【核心知识点：将 RAG 变成 Tool】
    用于查询公司的各类规章制度（如考勤、年假、报销、福利等）。
    参数 query 必须是你提炼出的核心关键词。
    """
    print(f"\n  [RAG 检索中...] Agent 决定使用关键词 '{query}' 检索公司知识库...")
    return mock_vector_retriever(query)

def main():
    print("=== 开始运行 Agentic RAG 演示 ===")
    
    llm = get_model("openai")
    tools = [company_policy_search]
    
    # 对应课程：4.3 定义中间件构建完整 AgenticRAG 系统
    # 这里底层的 create_agent 会自动利用 LangGraph 建立起“推理 -> 路由 -> 执行 -> 返回 -> 再次推理”的循环回路
    agent = create_agent(
        model=llm, 
        tools=tools,
        system_prompt=(
            "你是公司的人事助理。你可以闲聊，但如果用户问关于公司制度的问题，"
            "你必须调用 `company_policy_search` 工具去查阅知识库。如果查不到，就说不知道。"
        )
    )
    
    # 测试 1：闲聊（Agent 会决定不调用 RAG，直接回答）
    print("\n[测试1: 闲聊 - 期望不触发检索]")
    res1 = agent.invoke({"messages": [HumanMessage(content="你好，吃了吗？")]})
    print(f"AI: {res1['messages'][-1].content}")

    # 测试 2：业务问题（Agent 会决定提炼关键词并调用 RAG）
    print("\n[测试2: 业务问题 - 期望触发检索]")
    res2 = agent.invoke({"messages": [HumanMessage(content="我昨天晚上10点半才下班，打车的钱能报销吗？还有我刚入职半年，有年假吗？")]})
    print(f"\nAI: {res2['messages'][-1].content}")

    # 【补充说明：对应课程 4.4 LangSmith 可视化工具】
    # 要使用 LangSmith 监控上述 Agentic RAG 的执行轨迹（它到底搜了什么词、花了多少 Token），
    # 你只需在 `.env` 文件中添加：
    # LANGCHAIN_TRACING_V2=true
    # LANGCHAIN_API_KEY="你的_LANGSMITH_API_KEY"
    # LangChain 会在后台自动将数据异步上传到控制台画出可视化图表。

if __name__ == "__main__":
    main()
