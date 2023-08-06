import openai
import streamlit as st

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0px; color: red; font-size: 70px;'>ResponseBot</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center; margin-top: 0px;'>Your AI Assistant for Engaging Video Replies</h5>",
    unsafe_allow_html=True)

openai.api_key = "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"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Step 1: Ask the user to paste their comment
if prompt := st.text_area("Paste your comment and notes here!"):
    # Add the user's comment to the conversation context
    response_type = st.radio("Select response type:", ("Formal", "Informal"))
    num_words = st.number_input("Enter the number of words in the response:", min_value=10, max_value=200)

    st.session_state.messages.append({"role": "user", "content": f'''You are a Bot who generates response to comment for video creators. \
When video creator gives comment: {prompt}. The video creator wants response in {response_type} tone and in approximate {num_words} words limit'''})

    # Step 4: Generate the response
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
