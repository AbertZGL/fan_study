from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import importlib

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

def main():
    print("=== 开始运行 LCEL 高级特性: 失败回退机制 (Fallbacks) ===")

    # 1. 制造一个“坏”的模型
    # 我们故意设置一个不存在的模型名，它调用一定会报错（比如 API 挂了或网络不通）
    try:
        from langchain_openai import ChatOpenAI
        bad_llm = ChatOpenAI(model="gpt-99-not-exist", max_retries=0)
    except Exception:
        # 为了兼容性，如果你没装 openai，我们就换成其它一定报错的方式
        bad_llm = get_model("openai")
        bad_llm.model_name = "this-will-fail-12345"

    # 2. 准备一个“好”的模型（备用模型），例如：
    good_llm = get_model("openai")

    prompt = ChatPromptTemplate.from_template("解释一下什么是：{topic}")

    # 3. 构建主链（如果报错就整个失败）
    chain_that_fails = prompt | bad_llm | StrOutputParser()

    # 4. 构建带回退的链条（这是生产环境的重点！）
    # 【LangChain v1.0+ 核心知识点】：
    # 1. `with_fallbacks` 允许我们定义一串备用计划。当主模型遇到 RateLimitError/Timeout 等错误时，自动切换。
    # 2. 除了模型级别，你甚至可以在整个 Runnable 链条级别加 fallback。
    # 3. 这种基于 LCEL 内部实现的自动降级（如 OpenAI -> Anthropic -> 智谱），是提升应用鲁棒性的神器。
    chain_with_fallback = prompt | bad_llm.with_fallbacks([good_llm]) | StrOutputParser()

    print("\n[测试 1: 没有 Fallback，应该会报错并抛出异常]")
    try:
        chain_that_fails.invoke({"topic": "引力波"})
    except Exception as e:
        print(f"预期中的报错拦截成功：\n  -> {type(e).__name__}: {str(e)[:100]}...")

    print("\n[测试 2: 带有 Fallback，当 bad_llm 报错时，自动无缝切换 good_llm]")
    result = chain_with_fallback.invoke({"topic": "引力波"})
    
    print("\n[AI 最终成功回复（来自备用模型）]:")
    print(result)

if __name__ == "__main__":
    main()
