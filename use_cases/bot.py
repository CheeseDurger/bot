from typing import List, Sequence

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.agents import tool, AgentExecutor
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.docstore.document import Document
from langchain.document_loaders import WikipediaLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import MessagesPlaceholder
from langchain.chat_models import ChatOpenAI

from .document_retriever import DocumentRetriever

class Bot:
    _llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)
    agent: AgentExecutor

    @staticmethod
    @tool("search-the-web")
    def search(query: str) -> str:
        """useful for when you need to answer questions about current events"""
        content: str = DuckDuckGoSearchRun().run(query) + "\n"
        content += "Source : DuckDuckGo > " + query + "\n"
        content += "\n\n"
        return content

    @staticmethod
    @tool("get-documents-about-PUI")
    def get_pui_documents(query: str) -> str:
        """useful for when you need to answer questions about French Poles Universitaires d'Innovation (PUI)"""
        retriever: VectorStoreRetriever = DocumentRetriever().retriever()
        documents: List[Document] = retriever.invoke(query)
        content: str = ""
        for index, document in enumerate(documents):
            content += "### DOCUMENT " + str(index + 1) + " ###\n"
            content += document.page_content + "\n"
            content += "Source : " + document.metadata["source"] + "\n"
            content += "Page : " + str(document.metadata["page"]) + "\n"
            content += "\n\n"
        return content

    @staticmethod
    @tool("search-Wikipedia-encyclopedia")
    def get_wiki_content(query: str) -> str:
        """useful for when you need to answer encyclopedic questions"""
        documents: List[Document] = WikipediaLoader(query=query, load_max_docs=1).load()
        content: str = ""
        for index, document in enumerate(documents):
            content += "### DOCUMENT " + str(index + 1) + " ###\n"
            content += document.page_content + "\n\n"
            content += "Source : " + document.metadata["source"] + "\n"
            content += "\n\n"
        return content

    def __init__(self):
        tools: Sequence = [self.search, self.get_pui_documents, self.get_wiki_content]
        agent_kwargs = {"extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")]}
        memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
        self.agent: AgentExecutor = initialize_agent(
            tools,
            self._llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
            # verbose=True,
        )


if __name__ == "__main__":
    agent = Bot().agent
    while True:
        print("=> Question :")
        question: str = input()
        print("=> Réponse :")
        print(agent.run(question))
        print("\n")
