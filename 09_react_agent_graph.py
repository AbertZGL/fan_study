from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import importlib

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

@tool
def calculate_salary(base: int, bonus: int) -> int:
    """计算发薪总额。需要基础工资和奖金两个参数。"""
    print(f"  [工具执行中...] 计算 {base} + {bonus}")
    return base + bonus

@tool
def get_weather(location: str) -> str:
    """获取指定城市的天气预报。"""
    print(f"  [工具执行中...] 查询 {location} 天气")
    if "北京" in location:
        return "晴，20度，适合出门"
    else:
        return "多云，可能有小雨"

def main():
    print("=== 开始运行 LangGraph 进阶整合场景：ReAct 智能体 ===")
    
    # 1. 准备大模型实例
    llm = get_model("openai")
    
    # 2. 准备工具列表
    tools = [calculate_salary, get_weather]
    
    # 3. 创建记忆 Checkpointer
    memory = MemorySaver()
    
    # 4. 利用 langgraph 官方提供的高能构建封装，一行代码创建一个带有循环判断能力、自动调用工具能力的 Graph
    # 【LangGraph 最新知识点】：
    # 1. 在 LangGraph V1.0 之后，原先的 `create_react_agent` 被迁移到了 `langchain.agents.create_agent`！
    # 2. 彻底替代了 LangChain 原来的 `AgentExecutor`。
    # 3. 它背后自动构造了一个循环图：接收消息 -> 丢给模型 -> 如果模型说要用工具，就走工具节点并执行 -> 工具结果返回给模型重新思考 -> 直到不再调用工具则 END。
    # 4. 这不仅让执行过程更加透明可控，更允许你通过传入 `checkpointer` 实现记忆和随时中断。
    agent = create_agent(
        model=llm, 
        tools=tools, 
        checkpointer=memory
    )
    
    config = {"configurable": {"thread_id": "hr_session"}}

    # ======= 测试 1 =======
    print("\n[测试1：只回答信息，不适用工具]")
    for event in agent.stream(
        {"messages": [("user", "你好，我是新来的员工，请问怎么称呼你？")]},
        config=config,
        stream_mode="values"
    ):
        message = event["messages"][-1]
        message.pretty_print()


    # ======= 测试 2 =======
    print("\n[测试2：要求同时调用工具，计算工资和询问天气]")
    for event in agent.stream(
        {"messages": [("user", "我的基础工资是15000，加上3000奖金。这月能拿多少？拿着这钱周末去北京玩天气合适吗？")]},
        config=config,
        stream_mode="values"
    ):
        message = event["messages"][-1]
        # stream_mode="values" 会在每次图的 state 更新时把整个消息列表吐出来。我们只看最后一条。
        message.pretty_print()


if __name__ == "__main__":
    main()
