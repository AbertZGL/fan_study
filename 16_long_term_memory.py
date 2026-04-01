import importlib
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# MemorySaver 是一种检查点保存器 (Checkpointer)，但它只保存在内存中，重启即丢。
from langgraph.checkpoint.memory import MemorySaver

# ======================================================================
# 对应课程：5.4 检查点的特定实现类型 - SqliteSaver / PostgresSaver 等
# 在企业级生产环境中，我们不能用 MemorySaver，而是要用数据库将整个图的执行轨迹（Snapshot）持久化。
# 比如：
#   from langgraph.checkpoint.sqlite import SqliteSaver
#   import sqlite3
#   conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
#   memory = SqliteSaver(conn)
# 这个 memory 对象就像一个可以按 thread_id（会话ID）任意读写的大脑海马体，
# 即使重启服务器，只要拿着 thread_id，Agent 就能原地“复活”，连上下文都不丢。
# ======================================================================

# ======================================================================
# 对应课程：5.5 长期记忆和 Store（仓库）
# 
# 【最新知识点】：在 LangGraph 0.2.20+ 版本中，官方引入了全新的 `BaseStore` 概念！
# 为什么要有 Store？
# 1. 短期记忆 (Checkpointer)：它绑定在一个 `thread_id` 上。比如你昨天跟 AI 聊的 A 会话，
#    今天新开了一个 B 会话（换了 thread_id），AI 就会把昨天聊的完全忘光。
# 2. 长期记忆 (Store)：它跨越所有会话（Cross-Thread）。它就像一个真正的数据库。
#    比如：AI 记住“这个用户叫林客，他喜欢用 Python”，这不应该存放在某一个聊天会话里，
#    而是应该存放在一个全局的、属于用户的 Profile 命名空间（Namespace）里。
# ======================================================================
from langgraph.store.memory import InMemoryStore

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chat_model_node(state: State, config: dict, store: InMemoryStore):
    """
    一个极其强大的节点：它既能读取聊天上下文（通过 state），
    又能根据用户的专属 ID，从全局仓库（store）中读取该用户的长期档案。
    """
    llm = get_model("openai")
    
    # 1. 提取当前执行者的用户 ID
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    # 2. 从 Store 中获取长期记忆档案 (Namespace 为 ["profiles", user_id])
    profile_item = store.get(("profiles", user_id), "info")
    
    if profile_item:
        user_info = profile_item.value.get("preferences", "没有记录任何偏好")
    else:
        user_info = "第一次见面的陌生人"

    # 3. 动态组装系统 Prompt
    system_prompt = f"你是一个智能助手。请根据用户的长期档案来回答问题。\n[当前用户档案]: {user_info}"
    
    # 将系统提示词和对话历史发给大模型
    messages = [("system", system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    
    return {"messages": [response]}

def main():
    print("=== 开始运行 长期记忆(Store)与短期记忆(Checkpointer) 演示 ===")
    
    workflow = StateGraph(State)
    workflow.add_node("agent", chat_model_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    # 短期记忆：用于维持单次聊天的上下文
    checkpointer = MemorySaver()
    # 长期记忆：用于跨越聊天会话，存储用户画像、知识片段等全局数据
    store = InMemoryStore()

    # 在编译时，将 Store 和 Checkpointer 一起注入给图
    app = workflow.compile(checkpointer=checkpointer, store=store)

    # 模拟：我们在数据库后台（Store）中悄悄给 user_001 写了一份长期档案
    print("\n[后台] 正在往全局 Store 中写入 user_001 的偏好档案...")
    store.put(
        ("profiles", "user_001"), # Namespace 命名空间
        "info",                   # Key 键
        {"preferences": "用户喜欢代码极其简短，不要废话，最喜欢的语言是 Rust。"} # Value 值
    )

    # 测试 1：User 1 来聊天了
    print("\n[场景 1: user_001 开始聊天]")
    config_user1 = {"configurable": {"thread_id": "thread_abc", "user_id": "user_001"}}
    res1 = app.invoke(
        {"messages": [HumanMessage(content="帮我写个反转字符串的函数")]},
        config=config_user1
    )
    print(f"AI: {res1['messages'][-1].content}")

    # 测试 2：User 2 (陌生人) 来聊天了
    print("\n[场景 2: user_002 开始聊天]")
    config_user2 = {"configurable": {"thread_id": "thread_xyz", "user_id": "user_002"}}
    res2 = app.invoke(
        {"messages": [HumanMessage(content="帮我写个反转字符串的函数")]},
        config=config_user2
    )
    print(f"AI: {res2['messages'][-1].content}")

    # 结果显而易见：即使用户没有自我介绍，Agent 也能通过跨线程的 Store 读取到 user_001 的喜好（写了极简的 Rust 代码），
    # 而对 user_002 则是通用的礼貌回答（可能给出了详细解释的 Python 代码）。

if __name__ == "__main__":
    main()
