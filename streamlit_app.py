import json
from datetime import datetime
from xml.dom.minidom import Document

import openai
import streamlit as st

import database
from models.message import save_message
from models.result import save_result
from models.session import create_new_session
import models
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document



# Add custom CSS for right-to-left text styling
st.markdown(
    """
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def import_llm_models():
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                     model="gpt-4o",
                     temperature=0.5) #gpt-4o-2024-08-06 with fine tuning
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    return llm #, model

def load_character_prompt_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_character_prompt_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_initial_conversation(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if 'database_initialized' not in st.session_state:
    database.create_database()
    st.session_state.database_initialized = True

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = create_new_session("Chat Session Name")

if 'chat_initialized' not in st.session_state:
    # connect openai key
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.llm = import_llm_models()

    # st.session_state.character_prompt = load_character_prompt_txt("character_prompt.txt")
    # st.session_state.messages = load_initial_conversation("initial_conversation.json")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"#"gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = load_character_prompt_json("character_prompt.json") #[]
        st.session_state.chat_start_index = len(st.session_state.messages) - 1
    st.session_state.chat_initialized = True

if 'page' not in st.session_state:
    st.session_state.page = "Home"  # Default page is Home

def page_chat():
    st.title("מוקד רפואה מרחוק")
       # Add styled medical record section
    st.markdown(
        """
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px;direction: rtl; text-align: right;">
            <strong>תיק רפואי של מר. יונתן בניון:</strong> <br>
            <strong>COPD מתקדם:</strong> Prednisolone 10 mg, Fluticasone inhaler 500 mcg, חמצן <br>
            <strong>יתר לחץ דם:</strong> Amlodipine 5 mg, Furosemide 40 mg <br>
            <strong>סוכרת סוג 2:</strong> Novorapid <br>
            <strong>היסטוריה של עישון כבד:</strong> 40 שנות קופסא, הפסיק לעשן לפני 5 שנים
        </div>
        """,
        unsafe_allow_html=True
    )
    #st.markdown("**:תיק רפואי של מר. יונתן בניון**  \n**COPD מתקדם:** Prednisolone 10 mg, Fluticasone inhaler 500 mcg, חמצן  \n**יתר לחץ דם:** Amlodipine 5 mg, Furosemide 40 mg  \n**סוכרת סוג 2:** Novorapid  \n**היסטוריה של עישון כבד:** 40 שנות קופסא, הפסיק לעשן לפני 5 שנים.")
       # home_button = st.button("Finish chat", icon=":material/send:")
    # if home_button:
    #     st.session_state.page = "Result"
    #     st.rerun()
    
  #st.markdown("<br><br><br><br>", unsafe_allow_html=True", unsafe_allow_html=True)  # Add a line break after the medical record section


    for message in st.session_state.messages[st.session_state.chat_start_index:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # [-1]["text"]

    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        current_time = datetime.now()
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "from": "assistant",
            "timestamp": current_time.isoformat()
        })
        # st.write("debug:", st.session_state.messages[-1])
        # st.write(st.session_state)
        save_message(
            "user",prompt,st.session_state.user_name,
            "assistant",current_time,st.session_state.user_email,
            st.session_state['session_id']
        )
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate stream of response with milliseconds delay
            # for response in openai.ChatCompletion.create(
            #         model=st.session_state["openai_model"],
            #         messages=[
            #                      # {"role": "system", "content": st.session_state.character_prompt}
            #                  ] + st.session_state.messages,
            #         # will provide lively writing
            #         stream=True,
            # ):
            # get content in response
            # full_response += response.choices[0].delta.get("content", "")
            full_response = st.session_state.llm(st.session_state.messages).content
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        response_time = datetime.now()

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "from": st.session_state.user_name,
            "timestamp": response_time.isoformat()
        })
        save_message(
            "user",
            full_response,
            "assistant",
            st.session_state.user_name,
            response_time,
            st.session_state.user_email,
            st.session_state['session_id'])
        # st.write(st.session_state)

    home_button = st.button("Finish chat", icon=":material/send:")
    if home_button:
        st.session_state.page = "Result"
        st.rerun()


def page_home():
    st.title("ברוכים הבאים לסימולטור וירטואלי")
    st.markdown("""
    אנא הזינו את **ארבעת הספרות האחרונות** של תעודת הזהות שלכם.  
    לאחר מכן, ייפתח חלון ובו תוכלו לנהל שיחה עם מטופל הפונה לעזרה באמצעות **מוקד של רפואה מרחוק**.  

    ### המשימה שלכם:
    - להבין את מצבו הרפואי של המטופל.  
    - לבצע אומדנים ולקבל החלטות.  
    - להקשיב למטופל ולשאול שאלות.  

    **בהצלחה!**
    """)
    #st.write(""" ☏ אנא הזינו את ארבעת הספרות האחרונות של תעודת הזהות שלכם. לאחר מכן, ייפתח חלון ובו תוכלו לנהל שיחה עם מטופל הפונה לעזרה באמצעות מוקד של רפואה מרחוק. המשימה שלכם היא להעניק היא להבין את מצבו הרפואי, לבצע אומדנים ולקבל החלטות. הקשיבו למטופל, שאלו אותו שאלות וקבלו החלטות בהתאם. בהצלחה!  """)
    user_name = st.text_input("ארבע ספרות אחרונות של תעודת זהות")
    chat_button = st.button("הקליקו כדי להתחיל בסימולציה")
    if chat_button and user_name:
        user_email = f"{user_name.strip()}@test.cop"
        new_user = models.user.User(name=user_name, email=user_email)
        if 'user_added' not in st.session_state:
            models.user.add_user(new_user, user_email)
            st.session_state.user_added = True
        st.session_state.user_name = user_name
        st.session_state.user_email = user_email

        st.session_state.page = "Chat"
        st.rerun()


def page_result():
    st.title("Summarize")
    st.write("This is the Result page.")
    summarize = summarize_chat()
    st.write(summarize)
    result_time = datetime.now()
    save_result(summarize, result_time, st.session_state.user_email, st.session_state['session_id'])

def llm_page_result():
    st.title("Summarize")
    st.write("This is the Result page.")
    summary = llm_summarize_conversation()
    st.write(summary)
    result_time = datetime.now()
    save_result(summary, result_time, st.session_state.user_email, st.session_state['session_id'])

def llm_summarize_conversation():

    # You can provide the full conversation history as input to the summarizer
    # full_conversation = "\n".join([msg["text"] for msg in st.session_state.messages])#  if isinstance(msg, (HumanMessage, AIMessage))
    full_conversation = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[st.session_state.chat_start_index:]
    )
    docs = [Document(page_content=full_conversation)]

    summarize_chain = load_summarize_chain(llm=st.session_state.llm, chain_type="stuff")
    return summarize_chain.run(docs)

def summarize_chat():
    if len(st.session_state.messages) == 0:
        return "No conversation to summarize."

    # Concatenate the chat history
    chat_history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[st.session_state.chat_start_index:]
    )

    # st.write("Messages:", *st.session_state.messages[st.session_state.chat_start_index:])
    summary_prompt = [
        SystemMessage(content="Summarize the following conversation."),
        *st.session_state.messages[st.session_state.chat_start_index:]
    ]
    response = st.session_state.llm(summary_prompt).content

    # Use OpenAI to summarize the chat
    # response = openai.ChatCompletion.create(
    #     model="gpt-4o",#"gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
    #         {"role": "user", "content": f"Please summarize the following conversation:\n{chat_history}"}
    #     ]
    # )
    # st.write('Debug: ', response)
    summary = response #['choices'][0]['message']['content']
    return summary


# page = st.radio("Choose a page", ("home", "Chat", "Result"))
# Display the corresponding page
if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Chat":
    page_chat()
elif st.session_state.page == "Result":
    llm_page_result()
