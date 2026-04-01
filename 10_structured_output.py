from pydantic import BaseModel, Field
from typing import List
import importlib

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

# 1. 定义我们期望大模型返回的严格数据结构
class CharacterInfo(BaseModel):
    name: str = Field(description="角色的全名")
    age: int = Field(description="角色的年龄")
    traits: List[str] = Field(description="角色的性格特征列表，至少3个")

class StoryAnalysis(BaseModel):
    main_character: CharacterInfo = Field(description="故事的主角信息")
    summary: str = Field(description="一句话总结故事梗概")
    is_happy_ending: bool = Field(description="这是一个完美的结局吗？")

def main():
    llm = get_model("openai")

    # 【LangChain v1.0+ 核心知识点】：
    # 1. `with_structured_output` 是目前强制模型输出特定 JSON/对象的绝对最佳实践。
    # 2. 它彻底淘汰了以前手写 prompt ("请返回如下格式的JSON：...") 以及脆弱的正则解析(PydanticOutputParser)。
    # 3. 只要底层模型支持 Tool Calling（如 GPT-4o, Claude 3, GLM-4 等），它会直接将 Pydantic 类转化为 Schema 并强制模型输出该结构。
    structured_llm = llm.with_structured_output(StoryAnalysis)

    story = """
    在很久以前的硅谷，有一位名叫林客（Link）的资深程序员，今年35岁。
    他平时沉默寡言，但写代码时极为专注，且非常乐于助人。
    某天他终于修完了一个存在了十年的祖传Bug，公司决定给他放一个月带薪假。
    他开心地买了一张去夏威夷的机票，躺在沙滩上喝着椰汁。
    """

    print("=== 开始运行 Structured Output 强制结构化输出 ===")
    print("正在阅读故事并提取对象化数据...")

    # 返回的结果不再是字符串，而是一个原生的 Python Pydantic 对象！
    result: StoryAnalysis = structured_llm.invoke(story)

    print("\n[提取结果 - 原生Python对象]:")
    print(f"主角姓名: {result.main_character.name}")
    print(f"主角年龄: {result.main_character.age}")
    print(f"性格特征: {', '.join(result.main_character.traits)}")
    print(f"故事总结: {result.summary}")
    print(f"是否Happy Ending: {'是' if result.is_happy_ending else '否'}")

if __name__ == "__main__":
    main()
