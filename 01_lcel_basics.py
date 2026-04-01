from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import importlib

# 导入刚才你配置的多模型实例获取函数（由于文件名不能作为python标准模块导入，我们用 importlib 动态引入）
setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

def main():
    # 1. 初始化模型 （这里我们使用 zhipu, 或者 openai 根据你在 env 里的配置）
    # 如果想用别的，比如 'openai' 获取其他平台，请在调用时传入 'openai', 'deepseek', 'anthropic' 等。
    llm = get_model("openai") 

    # 2. 构建 PromptTemplate
    # Langchain推荐使用 ChatPromptTemplate 来处理系统和用户的角色
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个讲究效率的得力助手。请简洁明了地回答用户问题。"),
        ("user", "{question}")
    ])

    # 3. 构建 OutputParser
    # 把模型的原始返回体（AIMessage对象）解析为普通的字符串(String)
    output_parser = StrOutputParser()

    # 4. 构建 LCEL 链条 (LangChain Expression Language)
    # 【LangChain v0.2/v0.3 最新知识点】：
    # 1. LCEL 是当前 LangChain 推荐和唯一的标准链条构建方式，之前的 `LLMChain` 已被弃用（Deprecated）。
    # 2. `|` (管道符号) 被重载以组合 `Runnable` 协议的对象，这意味着所有的 LCEL 组件（Prompt, Model, OutputParser）都实现了相同的接口（`invoke`, `stream`, `batch`）。
    # 3. 数据流：输入的字典 -> prompt(格式化模板，返回PromptValue) -> llm(生成原始回复AIMessage) -> output_parser(提取字符串)
    chain = prompt | llm | output_parser

    # 5. 调用执行
    # 【LangChain v0.2/v0.3 最新知识点】：
    # `invoke` 替代了旧版本的 `__call__` 或 `run`。所有 `Runnable` 组件都暴露了 `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch` 等标准方法。
    print("=== 开始运行 LCEL 基础测试 ===")
    response = chain.invoke({"question": "讲一个笑话"})
    print("\n[AI 的回答]:")
    print(response)

if __name__ == "__main__":
    main()
