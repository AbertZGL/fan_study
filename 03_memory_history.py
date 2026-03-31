from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
import importlib

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

# 用于模拟数据库存储各个用户的历史记录
store = {}

def get_session_history(session_id: str):
    # 根据 session_id 获取对话历史列表（如果不存在则创建一个全新的）
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def main():
    llm = get_model("openai")

    # 注意：MessagesPlaceholder 用来标明历史对话消息插入的位置
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的聊天机器人。"),
        MessagesPlaceholder(variable_name="history"), 
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    # 通过 RunnableWithMessageHistory 将我们的 chain 包裹起来
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question", # 告诉它哪个键是用户的新问题
        history_messages_key="history", # 告诉它哪个键放历史消息
    )

    print("=== 开始带有上下文记忆的对话 ===")
    
    # Session 1: 第一轮对话
    print("\n[Session 1 - 回合 1] User: 嗨，我是一名前端工程师，我叫小明。")
    response1 = chain_with_history.invoke(
        {"question": "嗨，我是一名前端工程师，我叫小明。"},
        config={"configurable": {"session_id": "session-123"}}
    )
    print("AI:", response1)

    # Session 1: 第二轮对话，测试模型是否记住了名字和职业
    print("\n[Session 1 - 回合 2] User: 你能推荐一个我应该学的框架吗？顺便猜猜我叫啥。")
    response2 = chain_with_history.invoke(
        {"question": "你能推荐一个我应该学的框架吗？顺便猜猜我叫啥。"},
        config={"configurable": {"session_id": "session-123"}}
    )
    print("AI:", response2)

    # Session 2: 换一个用户 session
    print("\n[Session 2 - 陌生人测试] User: 你知道我的职业吗？")
    response3 = chain_with_history.invoke(
        {"question": "你知道我的职业吗？"},
        config={"configurable": {"session_id": "session-456"}}
    )
    print("AI:", response3)

if __name__ == "__main__":
    main()
