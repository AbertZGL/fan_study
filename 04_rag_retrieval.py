from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings # 这里统一使用 OpenAI Embeddings，也可以换成本地 HuggingFace
import importlib
import os

setup_module = importlib.import_module("00_multi_model_setup")
get_model = setup_module.get_model

def main():
    llm = get_model("openai")

    # --- 1. 数据准备 (模拟本地产生一份知识库文档) ---
    fake_kb_path = "temp_knowledge.txt"
    with open(fake_kb_path, "w", encoding="utf-8") as f:
        f.write("公司的 WiFi 密码是：FanStudyAI-2026。\n")
        f.write("周末加班可以报销100元打车费。\n")
        f.write("年假规则是入职满一年即可获得5天年假。\n")
    
    print("=== 开始运行 RAG (Retrieval-Augmented Generation) 流程 ===")
    
    # --- 2. 文档加载与切分 ---
    loader = TextLoader(fake_kb_path, encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    # --- 3. 向量化与存储 (Embeddings & VectorStores) ---
    # 如果你没有 OpenAI api key 的向量模型使用权限，也可以换为 HuggingFaceEmbeddings()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    # 将向量数据库转化为 Retriever（检索器）
    # 当调用 retriever.invoke("问题") 时，它会返回相关的切片文档
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # --- 4. 构建回答 Prompt 和 LCEL Chain ---
    template = """基于以下提供的上下文（Context）来回答用户的问题。如果上下文中没提到，请回答不知情。

Context: {context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        # 将多个 document 对象拼成一段纯文本给大模型阅读
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG chain 完整链路：
    # 并行第一步： 
    #   context 字段 -> 使用 retriever 检索出文档 -> 使用 format_docs 拼成字符串
    #   question 字段 -> RunnablePassthrough() 透传原字符串
    # 第二步：将这两个输入注入 prompt
    # 第三步：送入 llm
    # 第四步：解析为字符串输出
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 5. 执行和测试 ---
    question = "我要连WiFi，密码是多少？"
    print(f"\nUser: {question}")
    print(f"AI: {rag_chain.invoke(question)}")

    question2 = "周末加班有啥福利吗？"
    print(f"\nUser: {question2}")
    print(f"AI: {rag_chain.invoke(question2)}")

    # 清理临时文件
    if os.path.exists(fake_kb_path):
        os.remove(fake_kb_path)

if __name__ == "__main__":
    main()
