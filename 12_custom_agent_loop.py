import importlib
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

# --- 定义工具 ---
@tool
def get_stock_price(ticker: str) -> str:
    """查询股票的最新价格。如果被询问具体某支股票，使用此工具。"""
    print(f"\n  >> [工具调用] 查询 {ticker} 的股价...")
    if ticker == "AAPL":
        return "150.00"
    return "未知代码"

# --- 定义状态 ---
# 【LangGraph V1.0+ 核心知识点】：
# 1. 状态里必须有 messages 列表，且带有 add_messages 更新策略。
class State(TypedDict):
    messages: Annotated[list, add_messages]

def main():
    print("=== 开始运行 手搓底层 Agent (剖析 create_agent 的本质) ===")
    
    # 1. 准备大模型，并绑定工具（这步是必须的，赋予模型理解工具结构的能力）
    tools = [get_stock_price]
    llm = get_model("openai").bind_tools(tools)

    # 2. 定义 Graph 节点函数：调用模型
    def call_model(state: State):
        # 把之前的消息历史喂给模型，获得它新产生的消息（可能是聊天，也可能是要求调工具）
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # 3. 定义 Graph 节点函数：执行工具
    # 【LangGraph V1.0+ 核心知识点】：
    # 1. `ToolNode` 是一个官方提供的预置节点，它会自动扫描 state["messages"] 里最后一条消息。
    # 2. 如果里面有 `tool_calls`，它就会根据我们传入的 tools 列表，执行相应的 Python 函数。
    # 3. 执行完毕后，它会自动把结果包装成 `ToolMessage` 并追加到状态列表中。
    tool_node = ToolNode(tools)

    # 4. 构建核心状态图
    workflow = StateGraph(State)

    # 添加两个节点
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # 5. 连接边：定义执行流程
    # 开始必定先去让大模型思考
    workflow.add_edge(START, "agent")

    # 【LangGraph V1.0+ 核心知识点】：
    # 1. `tools_condition` 是一个官方内置的条件路由函数。
    # 2. 它会检查最后一条消息：如果含有 `tool_calls` 属性，它返回字符串 "tools"，否则返回 "__end__"。
    # 3. 这种自动化的条件判断，彻底免去了我们手写 `if "tool_calls" in msg.additional_kwargs:` 的烦恼。
    workflow.add_conditional_edges(
        "agent",          # 判断起点是 agent 节点
        tools_condition   # 官方的判断逻辑
    )

    # 工具执行完后，永远回到 agent 节点去重新思考，看是不是还要调别的工具，或者可以直接总结回答了
    workflow.add_edge("tools", "agent")

    # 6. 编译成 Runnable 图
    app = workflow.compile()

    # 测试它
    print("\n[测试：苹果公司的股票今天多少钱？]")
    initial_input = {"messages": [HumanMessage(content="苹果公司(AAPL)的股票今天多少钱？")]}
    
    # stream 迭代运行图中的每一个节点，方便观察执行轨迹
    for chunk in app.stream(initial_input, stream_mode="values"):
        # 打印当前所有消息的最后一条（最新产生的状态）
        last_msg = chunk["messages"][-1]
        print(f"[{last_msg.__class__.__name__}]: {last_msg.content[:50]}")

    print("\n[运行完成：你看，上面的逻辑其实就是 create_agent() 的底层实现源码！]")

if __name__ == "__main__":
    main()
