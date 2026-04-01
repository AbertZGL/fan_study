from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import importlib

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

# 【LangGraph 最新知识点】：
# 1. `add_messages` 是 LangGraph 官方提供的一个 Reducer 函数。
# 2. `Annotated[list, add_messages]` 告诉 LangGraph 这个变量是一个列表，当向列表中添加新元素时，
#    它不仅是覆盖，而是智能附加(append)或根据 message ID 进行合并更新。这非常适合聊天记录和工具调用结果的更新。
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chat_model_node(state: State):
    """一个仅仅调用模型进行闲聊的简单节点。它会读取 state 里的历史消息"""
    llm = get_model("openai")
    # 把用户输入及历史消息发给模型
    response = llm.invoke(state["messages"])
    # 按照规定，返回一个包含新 messages 的字典，通过 add_messages 合并逻辑，这个 AIMessage 会被追加到末尾
    return {"messages": [response]}


def main():
    print("=== 开始运行 LangGraph 中携带 Checkpointer 的持久状态保存图 ===")

    workflow = StateGraph(State)
    workflow.add_node("chat_model", chat_model_node)
    workflow.add_edge(START, "chat_model")
    workflow.add_edge("chat_model", END)

    # 核心：使用 MemorySaver。在生产中可以换为 PostgresSaver 或 SqliteSaver 等数据库持久化
    # 【LangGraph 最新知识点】：
    # 1. Checkpointer 是 LangGraph 实现 "Stateful" (有状态) 应用的核心机制。
    # 2. 每执行完一个节点，Checkpointer 就会将当前的 State 保存为一个 Snapshot（快照）。
    # 3. 这不仅实现了记忆（Memory），更重要的是它支持 "Time Travel" (时光倒流) 和 "Human-in-the-loop" (人类介入审批)。
    memory = MemorySaver()

    # 编译时必须把 checkpointer 注入进去
    app = workflow.compile(checkpointer=memory)

    # 每当我们要带上状态执行时，需要给定 thread_id（线程ID/会话ID/用户ID）
    config = {"configurable": {"thread_id": "user_id_001"}}

    print("\n[第一轮对话开始]")
    user_input = "你好！我叫阿强，我是一名前端工程师。"
    print(f"User: {user_input}")
    
    # 模拟输入流并追加用户消息
    # state["messages"] 中我们放一条 HumanMessage ("user")
    events = app.invoke({"messages": [("user", user_input)]}, config=config)
    # 取出最后一个消息（即刚刚那轮由大模型节点生成的消息内容）
    print(f"AI: {events['messages'][-1].content}")


    print("\n[第二轮对话开始] (在同一线程下)")
    user_input2 = "你能用一句话总结一下我们刚才的寒暄吗？以及我是谁？"
    print(f"User: {user_input2}")
    
    events2 = app.invoke({"messages": [("user", user_input2)]}, config=config)
    print(f"AI: {events2['messages'][-1].content}")
    
    print("\n[查看检查点存储]")
    # 你甚至可以追溯历史：
    checkpoint_tuples = list(memory.list(config))
    print(f"当前 config 下我们共有 {len(checkpoint_tuples)} 个版本的状态快照记录。")

if __name__ == "__main__":
    main()
