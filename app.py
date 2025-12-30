import streamlit as st
from rag_pipeline import answer_question

st.set_page_config(
    page_title="Northwind Benefits Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Northwind Benefits Q&A")
st.caption("Ask questions about Standard and Health Plus benefits")

# -------------------------
# Session state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    show_debug = st.checkbox("Show routing & retrieved context", value=False)
    clear = st.button("🧹 Clear chat")

    if clear:
        st.session_state.messages = []
        st.rerun()

# -------------------------
# Chat history
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat input (LOOP)
# -------------------------
user_query = st.chat_input("Ask a question about Northwind benefits...")

if user_query:
    # show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # run RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = answer_question(user_query)

        st.markdown(result["answer"])

        if show_debug:
            with st.expander("🔍 Debug details"):
                st.markdown(f"**Routed to:** `{result['route']}`")
                st.markdown("**Retrieved context:**")
                st.code(result["context"])

    # store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })
