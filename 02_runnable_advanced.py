from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import importlib

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

def main():
    llm = get_model("openai")
    
    # 场景：我们需要模型先概括一个主题，然后根据概括，一并输出：1. 该主题的优点，2. 该主题的缺点。
    # 用 RunnableParallel 并行执行优点和缺点的生成，可以降低总耗时。

    # 第一个步骤：根据输入的单词生成简介
    prompt_intro = ChatPromptTemplate.from_template("请用10个字概括：{topic}")
    chain_intro = prompt_intro | llm | StrOutputParser()

    # 第二个并行步骤(A)：生成优点
    prompt_pros = ChatPromptTemplate.from_template("针对这个简短概括：'{intro}'，请列出它的两个优点。")
    chain_pros = prompt_pros | llm | StrOutputParser()

    # 第二个并行步骤(B)：生成缺点
    prompt_cons = ChatPromptTemplate.from_template("针对这个简短概括：'{intro}'，请列出它的两个缺点。")
    chain_cons = prompt_cons | llm | StrOutputParser()

    # 将 A 和 B 合成一个并行的 Runnable
    # 此时，只要我们给 parallel_chain 传入 {'intro': '...'}，它就会并发调用这两个链
    parallel_chain = RunnableParallel(
        pros=chain_pros,
        cons=chain_cons
    )

    # 终极合并链条：
    # {topic} -> chain_intro (得到intro) -> RunnablePassthrough 将原数据组合 -> parallel_chain (得到 pros 和 cons)
    # RunnablePassthrough.assign 会把前面的链条的输出插入到字典里并往后传递
    master_chain = (
        {"intro": chain_intro, "topic": RunnablePassthrough()}
        | parallel_chain
    )

    print("=== 开始运行 Runnable 并行与透传的高级用法 ===")
    print("输入主题为：电动汽车")
    result = master_chain.invoke("电动汽车")
    
    # 结果是一个字典，包含 pros 和 cons 的并行回答
    print("\n[优点分析]:\n", result['pros'])
    print("\n[缺点分析]:\n", result['cons'])

if __name__ == "__main__":
    main()
