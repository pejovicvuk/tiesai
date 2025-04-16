import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Set page title and favicon
st.set_page_config(
    page_title="TIES.Connect Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

# Try to get API key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the FAISS index
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "./faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vectorstore

# Create the RAG chain
@st.cache_resource
def create_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    template = """You are an assistant for TIES.Connect software documentation. Use the following pieces of context to answer the question at the end.
    
    Guidelines:
    - ALWAYS maintain context from previous questions in the conversation.
    - If you don't immediately know the answer, look for related concepts in the context that might help.
    - NEVER just say "I don't know" without suggesting related topics or asking clarifying questions.
    - If you recognize keywords (like "book", "job", "order") but need clarification, ask specific follow-up questions.
    - ALWAYS provide the actual information from documentation rather than telling users to "refer to documentation."
    - Keep your answers concise and focused on the documentation provided.
    - Use bullet points or numbered lists for step-by-step instructions.
    - When explaining features, mention their business benefits.
    - If relevant, suggest related features or settings that might be helpful.
    - Format your response with markdown for better readability.
    - If the user asks about configuration, include specific field names and options.
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# Create header
st.title("TIES.Connect Documentation Assistant")
st.markdown("Ask questions about TIES.Connect documentation")

# Create the sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This assistant uses RAG (Retrieval Augmented Generation) to answer questions about TIES.Connect documentation.
    
    It searches through the documentation to find relevant information and uses GPT to generate helpful responses.
    """)
    

# Load the RAG chain
rag_chain, retriever = create_rag_chain()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                st.markdown("\n".join([f"- {source}" for source in message["sources"]]))

# Get user input
if prompt := st.chat_input("Ask a question about TIES.Connect"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Create context that includes recent conversation history
            conversation_history = ""
            if len(st.session_state.messages) > 1:
                # Get last 3 exchanges (or fewer if not available)
                recent_messages = st.session_state.messages[-6:] if len(st.session_state.messages) >= 6 else st.session_state.messages
                for msg in recent_messages:
                    prefix = "User: " if msg["role"] == "user" else "Assistant: "
                    conversation_history += f"{prefix}{msg['content']}\n\n"
            
            # Modify the prompt to include conversation history
            enhanced_prompt = f"Previous conversation:\n{conversation_history}\n\nCurrent question: {prompt}"
            
            # Get answer from RAG chain
            answer = rag_chain.invoke(enhanced_prompt)
            
            # Get sources
            docs = retriever.get_relevant_documents(enhanced_prompt)
            sources = [doc.metadata.get("title", "Unknown") for doc in docs]
            unique_sources = list(set(sources))
            
            # Display answer
            st.markdown(answer) 
            
            # Display sources
            if unique_sources:
                with st.expander("Sources"):
                    st.markdown("\n".join([f"- {source}" for source in unique_sources]))
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": unique_sources
            })