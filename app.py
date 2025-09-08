import streamlit as st
import requests

st.set_page_config(page_title="HR Resource Query Bot", page_icon="💼")

st.title("💼 HR Resource Query Bot")
st.write("Ask me to find employees based on skills, experience, and availability.")

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/chat"

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
query = st.text_input("Enter your query", placeholder="Find Python developers with 3+ years experience")

if st.button("Search") and query.strip():
    try:
        response = requests.post(API_URL, json={"query": query, "top_k": 3})
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            matches = data["matches"]

            st.session_state.history.append({"user": query, "bot": answer})

            st.subheader("💡 Recommendation")
            st.write(answer)

            st.subheader("📋 Matching Employees")
            for emp in matches:
                st.markdown(f"""
                **{emp['name']}**  
                - Skills: {", ".join(emp['skills'])}  
                - Experience: {emp['experience_years']} years  
                - Projects: {", ".join(emp['projects'])}  
                - Availability: {emp['availability']}
                """)
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

# Show chat history
if st.session_state.history:
    st.subheader("🗨 Chat History")
    for chat in reversed(st.session_state.history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")