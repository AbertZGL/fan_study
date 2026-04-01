import asyncio
import importlib
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

@tool
def heavy_computation(num: int) -> int:
    """模拟一个非常耗时的计算工具"""
    import time
    print(f"\n[后台系统] 正在计算 {num} 的平方，请稍等...")
    time.sleep(2)
    return num * num

async def main():
    print("=== 开始运行 V2 Async Streaming API: astream_events ===")
    
    # 【LangChain v1.0+ 核心知识点】：
    # 1. 在复杂的 Agent 或者 LCEL 链条中，直接用 `.stream()` 只能拿到最终结果的字符串块。
    # 2. 但很多时候，我们希望在前端显示“正在调用工具...”、“工具返回了结果...”、“AI 正在思考...”。
    # 3. 官方推出了终极杀器：`.astream_events(..., version="v2")`。
    # 4. 它能把整个执行图中的*每一个细微动作*（例如: 模型开始生成、生成一个 token、调用工具、工具结束）都以事件的形式抛出来。

    llm = get_model("openai")
    tools = [heavy_computation]
    
    # 我们用官方推荐的封装方法，快速创建一个带有工具调用能力的 Agent
    agent_executor = create_agent(llm, tools)

    question = "请帮我计算一下 999 的平方是多少？然后再给我讲个很短的关于数字的笑话。"

    # 启动异步事件流 (这是在真实业务（比如 FastAPI 或 WebSocket 中最常用的方式）
    # 注意必须带 version="v2"
    events = agent_executor.astream_events(
        {"messages": [HumanMessage(content=question)]},
        version="v2"
    )

    print("\n[开始监听底层执行事件流]:\n")
    
    # 【LangChain v1.0+ 核心知识点】：
    # 遍历抛出的每一个微小事件，根据 event 的类型进行 UI 渲染。
    async for event in events:
        kind = event["event"]
        
        # 1. 当模型开始逐字输出回答时
        if kind == "on_chat_model_stream":
            # 提取流式输出的 token
            content = event["data"]["chunk"].content
            if content:
                # 打印出模型吐出来的文字，不要换行
                print(content, end="", flush=True)

        # 2. 当发现大模型决定要调用工具时
        elif kind == "on_tool_start":
            tool_name = event["name"]
            tool_args = event["data"].get("input")
            print(f"\n[前端提示] -> AI 决定使用工具 '{tool_name}'，参数: {tool_args}...")

        # 3. 当工具执行完毕时
        elif kind == "on_tool_end":
            tool_name = event["name"]
            tool_result = event["data"].get("output")
            print(f"\n[前端提示] -> 工具 '{tool_name}' 执行完毕，结果是: {tool_result}")

    print("\n\n[事件流全部结束！]")

if __name__ == "__main__":
    # 因为涉及到异步方法 astream_events，我们需要用 asyncio 运行
    asyncio.run(main())
