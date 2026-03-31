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
    # LCEL 的核心使用 `|` （管道）符号连接。这和 Linux 里的管道一样。
    # 数据流： 输入的字典 -> prompt(格式化模版) -> llm(生成原始回复) -> output_parser(提取字符串)
    chain = prompt | llm | output_parser

    # 5. 调用执行
    print("=== 开始运行 LCEL 基础测试 ===")
    response = chain.invoke({"question": "请用一句话解释什么是大语言模型（LLM）？"})
    print("\n[AI 的回答]:")
    print(response)

if __name__ == "__main__":
    main()
