import vertexai
import openai
import streamlit as st

from google.auth import default, transport

# TODO(developer): Update and un-comment below line
PROJECT_ID = "test-nov-2024"
location = "us-central1"

# Programmatically get an access token
creds, project = default() #google.auth.#
auth_req = transport.requests.Request()#google.auth.#
creds.refresh(auth_req)
# Note: the credential lives for 1 hour by default (https://cloud.google.com/docs/authentication/token-types#at-lifetime); after expiration, it must be refreshed.

# Pass the Vertex endpoint and authentication to the OpenAI SDK
PROJECT_ID = PROJECT_ID
LOCATION = location

##############################
# Choose one of the following:
##############################

# If you are calling a Gemini model, set the MODEL_ID variable and set
# your client's base URL to use openapi.
MODEL_ID = 'gemini-1.5-flash'
client = openai.OpenAI(
    base_url = f'https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi',
    api_key = creds.token)

st.title("Chat mit Gemini")

if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = "google/gemini-1.5-flash"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model= st.session_state["gemini_model"],
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})