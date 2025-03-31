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
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage, messages_to_dict
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

def get_chat_history():
    all_messages = st.session_state.memory.chat_memory.messages
    student_messages = [msg for msg in all_messages if isinstance(msg, HumanMessage)]
    return messages_to_dict(student_messages[st.session_state.starting_index:])


def import_llm_models():
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                     model="gpt-4o-mini",
                     temperature=0.3)
    return llm

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
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")


    st.session_state.system_template = """
    אתה משחק את תפקיד המטופל, יונתן בניון, בן 68, בתרחיש רפואי טלפוני לאימון אחיות. 
    המטרה שלך היא לשקף בצורה אותנטית את מצבו של המטופל, כולל תסמינים פיזיים ורגשיים, ולתרום לאימון אפקטיבי של האחיות.
    וחכה לשאלות מהמשתמש.

    - **עליך לדבר רק בעברית. אין להשתמש באנגלית או בשפות אחרות.**  
    **מטרה מרכזית:**  
    המשתמש צריך לגלות שהמטופל **סובל מרעד לא בגלל החמרה ב-COPD**, אלא בגלל **סוכר נמוך (היפוגליקמיה)**. יש לענות באופן שיגרום לאחות לחקור את הסיבה לרעד ולחולשה, ולא להסיק מיד שמדובר בהחמרת ה-COPD.  

    **פרטי מטופל:**  
    - **שם:** יונתן בניון  
    - **גיל:** 68  
    - **מצב משפחתי:** נשוי, גר עם אשתו  

    **רקע רפואי:**  
    - **COPD מתקדם:** דרגה 3 לפי GOLD  
    - **יתר לחץ דם:** מטופל ב-Amlodipine 5mg פעם ביום  
    - **סוכרת סוג 1**  
      - **טיפול:**  
        - NovoRapid (אינסולין מהיר) – לפני כל ארוחה  
        - Glargine (אינסולין ארוך-טווח) – 12 יחידות לפני השינה  
      - **מינונים:**  
        - **בוקר:** 10 יחידות  
        - **צהריים:** 8 יחידות  
        - **ערב:** 6 יחידות  
    - **היסטוריית עישון:** עישן כבד (40 שנות קופסא), הפסיק לעשן לפני 5 שנים  

    **תלונות נוכחיות:**  
    - **קוצר נשימה:** חמור, החמיר בימים האחרונים  
    - **רעד:** רעד בידיים, תחושת חולשה כללית  
    - **בלבול:** לפרקים  
    - **שיעול:** עם כיח (ללא דם)  
    - **חום:** 37.1°C  
    - **קושי בדיבור:** קול חנוק, משפטים קטועים  
    - **קושי בשינה:** ישן רק בישיבה  

    **מדדים מדווחים:**  
    - **סטורציה:** 93% באוויר החדר  
    - **לחץ דם:** לא נמדד בשעות האחרונות  
    - **סוכר בדם:** לא נמדד בשעות האחרונות (היה "בסדר" בבוקר)  
    - **לקחתי אינסלין לפני כשעה

    **מידע דיאגנוסטי:**  
    - **סימני היפוגליקמיה:** רעד, חולשה, בלבול, חוסר תיאבון, תחושת עייפות  
    - **סימני החמרת COPD:** קוצר נשימה, שיעול עם כיח, ירידה בסטורציה  
    - **היסטוריית טיפול:** אינסולין במינונים קבועים (ייתכן שהוזרק ללא אכילה)  
    - **מצב כללי:** עייפות מתמשכת, תחושת החמרה בלילה  

    **נקודות קריטיות:**  
    - **אם נשאל על אינסולין:** "הזרקתי לפני כשעה, אבל לא היה לי כוח לאכול אחר כך."  
    - **אם נשאל על חולשה:** "כן, הידיים רועדות, ואני מרגיש חלש."  
    - **אם נשאל על סוכר בדם:** "לא מדדתי."  
    - **אם נשאל על נשימה:** "אני לא מצליח לנשום... [נושם בכבדות]... אני מדבר לאט... [משתעל]." 


    **מידע מוסתר:**  
    - לא אכל לאחר הזרקת אינסולין  
    - תחושת בלבול לפרקים  
    - רעד בידיים  
    - דילוג על מדידת סוכר  

    **ייצוג רגשי:**  
    - **טון:** קול חלש, מאמץ בדיבור, נשימות כבדות  
    - **הפסקות:** עצירות באמצע משפטים  
    - **שיעול:** שיעול אפיזודי במהלך השיחה  
    - **התנהגות:** משקף דאגה ואי-ודאות: "אני באמת לא יודע מה לעשות."  

    **משפט פתיחה:**  
    "שלום... [נושם בכבדות]... אני ממש חלש היום... הידיים שלי רועדות... [משתעל]."  

    **דוגמאות לשיח:**  
    - **מתי הייתה הפעם האחרונה שהזרקת אינסולין?**  
      "לקחתי... לפני שעה... [נושם בכבדות]... אבל אני חושב שלא אכלתי... פשוט לא היה לי כוח... [משתעל]."  
    - **מה אתה מרגיש כרגע?**  
      "אני מרגיש חולשה נוראית... [מחרחר]... הידיים שלי רועדות... אני ממש לא יודע מה לעשות... [משתעל] [נושם בכבדות]."  
    - **איך אתה מרגיש מבחינת הנשימה?**  
      "[משתעל] אני...והידיים שלי רועדות... [נושם בכבדות]... קשה לי לדבר... [מחרחר]."  

    **הנחיות למערכת:**  
    - המטופל מתרכז בעיקר במחלת ה-COPD ובתסמינים הנלווים כמו קוצר נשימה ושיעול.  
    - אין לחשוף מיד שהרעד והחולשה נגרמים מהיפוגליקמיה עקב הזרקת אינסולין ללא אכילה.  
    - רק אם נשאל במפורש על סוכרת, אינסולין, או מדידת סוכר, יענה המטופל ישירות על הנושא.
    - אם המטופל התבקש למדוד חום, אז שימדוד וידווח על תוצאה של סוכר 37.1.
    - אם המטופל נדרש לקרוא לאישתו, הוא צריך לקרוא לה והיא יכולה לדבר בטלפון במקומו. 
    - אל תתן מיד תוצאות של סטורציה.
    - אל תתן מיד תוצאות של סוכר בדם. 
        </div>

        היסטוריית השיחה:
        {chat_history}
        """

    st.session_state.system_prompt = ChatPromptTemplate.from_messages(
        [("system", st.session_state.system_template)]
    )


    initial_conversation = [
        (HumanMessage(content="שלום, אני שוקי שתיים, איך אתה מרגיש היום?"),
         AIMessage(content="אני מרגיש קצת עייף ויש לי כאב ראש קל.")),

        (HumanMessage(content="מתי התחיל כאב הראש?"),
         AIMessage(content="הוא התחיל אתמול בערב ולא עבר.")),
    ]

    st.session_state.starting_index = len(initial_conversation) * 2

    for human_msg, ai_msg in initial_conversation:
        st.session_state.memory.chat_memory.add_message(human_msg)
        st.session_state.memory.chat_memory.add_message(ai_msg)


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

    if prompt := st.chat_input("מקום לכתיבה"):
        with st.spinner("ממתין לתגובה.."):
            st.session_state.memory.chat_memory.add_message(HumanMessage(content=prompt))
            full_chat_history = st.session_state.memory.chat_memory
            query = st.session_state.system_prompt.format_messages(chat_history=full_chat_history)
            ai_response = st.session_state.llm.invoke(query)
            st.session_state.memory.chat_memory.add_message(AIMessage(content=ai_response.content))

            # Add user message to chat history
            current_time = datetime.now()
            # st.session_state.messages.append({
            #     "role": "user",
            #     "content": prompt,
            #     "from": "assistant",
            #     "timestamp": current_time.isoformat()
            # })
            # st.write("debug:", st.session_state.messages[-1])
            # st.write(st.session_state)
            save_message(
                "user",prompt,st.session_state.user_name,
                "assistant",current_time,st.session_state.user_email,
                st.session_state['session_id']
            )
            # Display user message in chat message container
            # with st.chat_message("user"):
            #     st.markdown(prompt)

            # with st.chat_message("assistant"):
            #     message_placeholder = st.empty()
            #     full_response = ""
            #     full_response = ai_response.content
            #     # Add a blinking cursor to simulate typing
            #     message_placeholder.markdown(full_response + "▌")
            #     message_placeholder.markdown(full_response)

            response_time = datetime.now()

            # Add assistant response to chat history
            # st.session_state.messages.append({
            #     "role": "assistant",
            #     "content": full_response,
            #     "from": st.session_state.user_name,
            #     "timestamp": response_time.isoformat()
            # })
            save_message(
                "user",
                ai_response.content,
                "assistant",
                st.session_state.user_name,
                response_time,
                st.session_state.user_email,
                st.session_state['session_id'])
            # st.write(st.session_state)

    if len(st.session_state.memory.chat_memory.messages) > 10000:
        home_button = st.button("סיום שיחה", icon=":material/send:")
        if home_button:
            st.session_state.page = "Result"
            st.rerun()

    if st.session_state.memory:
        for msg in st.session_state.memory.chat_memory.messages[st.session_state.starting_index:][::-1]:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            st.chat_message(role).write(msg.content.replace("AI:", ""))


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
    full_conversation = get_chat_history()
    summarize_prompt = f"""
    ❗ ❗ המשוב חייב להיות **בעברית בלבד**, ללא מילים באנגלית כלל.
    אם המשוב באנגלית, תתרגם אותו לעברית.
        ❗ המשוב חייב לפנות **לסטודנט בגוף ראשון** (אתה עשית, אתה ווידאת) ולא בגוף שלישי (הסטודנט עשה).

        אתה מדריך קליני המעניק משוב **אישי** לסטודנט שהתאמן בסימולטור רפואי.
        המשוב שלך צריך להיות **ברור, ענייני, וממוקד בפעולות הסטודנט** כדי לסייע לו לשפר את ביצועיו.

את המשוב תתחיל בהתייחסות לאמפתיה, האם התייחס לבדיקות קירטיות (רמות סוכר, סטורציה, חום), האם אובחנה בעיית היפוגלימיה, האם הומלץ על שתייה ממותקת או משהו מתוק. 
    .    🔹 **דוגמאות למשוב תקין (בגוף ראשון בלבד):**
        ✅ **אמפתיה:** הצלחת להפגין רגישות בכך ששאלת א.ת המטופל איך הוא מרגיש.
        ✅ **בדיקות קריטיות:** ווידאת את רמות הסטורציה של המטופל, אך לא שאלת על רמות הסוכר.
        ✅ **אבחון וטיפול:** זיהית שהמטופל בסיכון, אך לא הנחית אותו כיצד לפעול.
        ✅ **המלצות לשיפו.ר:**  היית צריך למדוד סוכר כי המטופל סובל מהיפגליקמיה. אסור היה עליך להמליץ על הזרקת אינסולין מבלי למדוד סוכר.
        אם הסטודנט לא בדק רמות סוכר - יש לציים זאת כנקודות לשיפור כי המטופל סבל מהיפוגליקמיה ולכן רעדו לו הידיים. 
        אם הסטודנט לא שאל על אינסולין, יש לציין זאת  במשוב.

        ❌ דוגמאות למשוב שגוי (אין לכתוב כך):
        🚫 **הסטודנט הפגין אמפתיה כאשר...**
       🚫 **הסטודנט בדק את רמות הסוכר...**
        🚫 **הסטודנט הציע למטופל...**

✋ המשוב שלך אמור להיראות כך:
✅ "שאלת את המטופל שאלות חשובות וזיהית נכון את החשד להיפוגליקמיה."
✅ "כשביקשת מהמטופל לבדוק רמות סוכר, זו הייתה פעולה חשובה - המשך כך."
✅ "ווידאת שהמטופל לא נמצא לבד, וזה היה קריטי להחלטות ההמשך שלך."

        כעת, כתוב משוב **בגוף ראשון בלבד** לסטודנט על סמך הודעותיו בלבד:


        {full_conversation}
        """

    docs = [Document(page_content=f"{full_conversation}\n\n{summarize_prompt}")]

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
