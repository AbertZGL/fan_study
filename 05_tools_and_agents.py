from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
import importlib

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

@tool
def multiply(a: int, b: int) -> int:
    """计算两个数字的乘积并返回"""
    return a * b

@tool
def get_weather(location: str) -> str:
    """获取指定城实的天气预报。如果不知道城市，就不应该调用。"""
    if "北京" in location:
        return "晴，20度，适合出门"
    else:
        return "多云，可能有小雨"

def main():
    llm = get_model("openai")

    # 1. 准备工具列表
    tools = [multiply, get_weather]
    
    # 【LangChain 最新知识点 (v1.0+)】：
    # 1. 历史终结：旧版 LangChain 的 `AgentExecutor`、`create_tool_calling_agent` 等类已经彻底从核心库中移除。
    # 2. 官方现在一律推荐使用基于 LangGraph 底层的 `create_agent` 方法。
    # 3. 以前需要手动写 `bind_tools`，还需要配 `AgentExecutor`；现在只需要传入 model 和 tools 即可。

    system_prompt = "你是一个拥有多种辅助工具的智能助理。遇到不会的问题或需要计算的功能，请优先调用工具。不要撒谎！"

    # 2. 创建预设引擎 (底层基于 LangGraph 的 Tool-Calling Agent)
    agent_executor = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt=system_prompt
    )

    print("=== 开始运行 LangChain Tools & Agent ===")
    
    # 示例1：调用乘法工具
    print("\n[测试1: 乘法]")
    # 注意输入格式的变化：由于底层变成了 Graph，现在的输入约定为 `{"messages": [{"role": "user", "content": "..."}]}`
    res1 = agent_executor.invoke({"messages": [{"role": "user", "content": "12乘以15等于多少？"}]})
    # 提取最后一条 AI 回复的消息
    print("[最终结果]:", res1['messages'][-1].content)
    
    # 示例2：调用天气工具
    print("\n[测试2: 天气]")
    res2 = agent_executor.invoke({"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]})
    print("[最终结果]:", res2['messages'][-1].content)


if __name__ == "__main__":
    main()
