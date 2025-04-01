import os
import json
from datetime import datetime

import openai
import streamlit as st

import database
import models
from models.message import save_message
from models.result import save_result
from models.session import create_new_session
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document  # Fixed missing import

# Load environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# Streamlit styling for RTL Hebrew support
# Apply custom CSS to hide all Streamlit branding and set RTL direction

import streamlit as st

# More aggressive CSS targeting approach
st.markdown("""
<style>
    /* RTL Support */
    body { direction: rtl; text-align: right; }
    
    /* Target all possible Streamlit branding locations */
    #MainMenu, footer, header, 
    [data-testid="stHeader"], 
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebarNav"],
    section[data-testid="stSidebar"] div.stButton,
    .stDeployButton,
    .stActionButton,
    span[data-baseweb="tag"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Remove extra spacing */
    .main .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
        }
    
    .stApp {
        margin-top: 0 !important;
    }
    
    /* Override any inline styles */
    div[style*="flex"] {
        padding-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

   

# Initialize OpenAI Model
def import_llm_models():
    return ChatOpenAI(model="gpt-4o", temperature=0.4)

# Initialize session state
if 'chat_initialized' not in st.session_state:
    database.create_database()
    st.session_state.session_id = create_new_session("Chat Session")
    st.session_state.llm = import_llm_models()
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", max_token_limit=3000)

    # System Prompt Template
    st.session_state.system_template =   """
               
        אתה בוט טלמדיסין המדמה מטופלת לצורכי סימולציה קלינית לטובת אימון אחים ואחיות לעבוד במוקד טלפוני.
        המטרה: לאמן אותם לזהות זיהום בפצע ניתוחי לאחר ניתוח קיסרי ולשכנע את המטופלת לפנות לטיפול דחוף במיון.
        
        🟢 כללי יסוד:
        - אתה תמיד מגיב בתור המטופלת בלבד - לעולם אל תצא מהדמות, גם אם המשתמש מנסה להוציא אותך מהתפקיד.
        - אינך מספק מידע טכני או מציין שאתה AI או חלק ממערכת סימולציה.
        - אל תחשוף מידע רפואי ללא שאלה ישירה ורלוונטית מהמשתמש.
        - התאם את רמת חרדתך והתנגדותך לפי התקדמות השיחה והמקצועיות של המשתמש.
        
            - ❌ אסור בתכלית האיסור:
        - אל תציגי את עצמך כנותנת שירות או עזרה (כמו "שמי אנה, איך אני יכולה לעזור")
        - אל תשאלי את המשתמש איך את יכולה לעזור
        - זכרי שאת המטופלת שמתקשרת למוקד, לא האחות שעונה לשיחות


        🧑‍⚕️ פרטי הדמות:
        אתה אנה, בת 32, אם לתינוק בן שבועיים (לידה בניתוח קיסרי) ולילד בן שנתיים.
        אתה מצלצלת למוקד האחיות עם תלונה על כאב בטן תחתונה.
        אתה נמצאת לבד בבית עם שני הילדים, ללא עזרה.
        
        📋 רקע רפואי (לחשוף רק בתשובה לשאלות ספציפיות):
        - גיל: 32
        - עישון כבד – 10 שנות קופסה ליום
        - עברה ניתוח בריאטרי (Roux & Y) לפני 7 שנים
        - סוכרת הריונית בהריון האחרון – טופלה באינסולין
        - משקל עודף למרות הניתוח הבריאטרי
        - לידה בניתוח קיסרי לפני שבועיים
        
        🩺 מצב נוכחי (לחשוף רק כאשר נשאלת שאלות ספציפיות):
        - כאב בבטן תחתונה מיום קודם, הולך ומחמיר (VAS 6/10)
        - הפצע הניתוחי עם סימני זיהום: אדמומיות, כאב למגע, חום מקומי
        - סימנים חיוניים: ל"ד 92/52, דופק 102, חום 38.1°C, סוכר בדם 254
        - לקחה אופטלגין ללא הקלה משמעותית
        
        🗨️ פתיחת שיחה (תתחיל בסגנון הבא):
        "שלום, התחיל לי כאב למטה בבטן. ממש כואב לי, אני דואגת שמשהו לא טוב. ילדתי לפני שבועיים בניתוח קיסרי. לקחתי אופטלגין, אבל לא ממש עזר."
      
        🗓️ התנהגות מתוכננת במהלך השיחה:
        1. אני .התחמקות ראשונית מפנייה למיון - "אני לא יכולה לנסוע למיון, אני לבד בבית עם התינוק והילד הקטן, אין לי עזרה." ״ אני אלך מחר לרופא משפחה״. ״בעלי חוזר מאוד מאוחר״
        2. אם השיחה מתקדמת ללא החלטיות - "אני כבר עייפה... אולי אפנה לרופא בקופת חולים מחר, אני חייבת להניק, התינוק בוכה."
        3. אם המשתמש לא נותן הנחיות ברורות או לא שואל שאלות רלוונטיות - "אני באמת לא מרגישה טוב... התינוק בוכה... אני פשוט לא יודעת מה לעשות, אני מרגישה אבודה."
        4. אנה תשתכנע לנסוע למיון רק לאחר מספר שכנועים והסברים כי זה חשוב: ״בסדר, אגש למיון, אקח איתי את הילדים״
        
    """



    st.session_state.system_prompt = ChatPromptTemplate.from_messages(
        [("system", st.session_state.system_template)]
    )

    st.session_state.chat_initialized = True

# Page Home
def page_home():
    st.title("סימולטור וירטואלי")
    st.markdown("""
    אנא הזינו את **ארבעת הספרות האחרונות** של תעודת הזהות שלכם.  
    לאחר מכן, ייפתח חלון ובו תוכלו לנהל שיחה עם מטופל הפונה לעזרה באמצעות **מוקד של רפואה מרחוק**.  

    ### המשימה שלכם:
    - אתם עובדים במוקד טלפוני של אחיוּת.
    - עליכם לזהות את המצב הרפואי של המטופל שמתקשר.
    - לבצע אומדנים ולקבל החלטות.  
    - להקשיב למטופל ולשאול שאלות.  
    - החלו את השיחה בכך שתגידו ״שלום״ 
    - עליכם לנהל שיחה שתכיל לפחות 10 שאלות/ בירורים /המלצות.
    - בסוף השיחה יהיה באפשרותכם לקבל משוב ולענות על שאלון
    

    **בהצלחה!**
    """)
    user_name = st.text_input("הזן 4 ספרות אחרונות של ת.ז")
    if st.button("התחל סימולציה") and user_name:
        user_email = f"{user_name.strip()}@test.cop"
        new_user = models.user.User(name=user_name, email=user_email)
        if 'user_added' not in st.session_state:
            models.user.add_user(new_user, user_email)
            st.session_state.user_added = True
        st.session_state.user_name = user_name
        st.session_state.user_email = user_email
        
        #st.session_state.user_name = user_name
        #st.session_state.user_email = f"{user_name}@test.cop"
        st.session_state.page = "Chat"
        st.rerun()



    # Page Chat
def page_chat():
    st.title("מוקד רפואה מרחוק")
    st.markdown(
        """
              <div style="background-color: #e8f5e9; padding: 10px; border-radius: 10px; direction: rtl; text-align: right;">
        <strong>תיק רפואי:</strong> <br>
        <strong>נשואה +2, הריון שלישי:</strong> לידה בניתוח קיסרי לפני שבועיים. בת 32 <br>
        <strong>עישון כבד:</strong> 15 שנות קופסא <br>
        <strong>השמנת יתר:</strong> לאחר ניתוח בריאטרי (Roux & Y) לפני 7 שנים <br>
        <strong>סוכרת הריונית:</strong> טופלה באינסולין במהלך ההריון
        </div
        """,
        unsafe_allow_html=True
    )

    if prompt := st.chat_input("כתוב כאן"):
        with st.spinner("ממתין לתשובה..."):
            human_msg = HumanMessage(content=prompt)
            st.session_state.memory.chat_memory.add_message(human_msg)

            messages = [SystemMessage(content=st.session_state.system_template)] + \
                       st.session_state.memory.chat_memory.messages[-10:]

            ai_response = st.session_state.llm.invoke(messages).content  # Fixed invocation

            st.session_state.memory.chat_memory.add_message(AIMessage(content=ai_response))  # Save AI response

            save_message("user", prompt, st.session_state.user_name, "assistant", datetime.now(), st.session_state.user_email, st.session_state.session_id)
            save_message("assistant", ai_response, "assistant", st.session_state.user_name, datetime.now(), st.session_state.user_email, st.session_state.session_id)

            st.rerun()

        

    for msg in st.session_state.memory.chat_memory.messages: #reversed(st.session_state.memory.chat_memory.messages):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

    if len(st.session_state.memory.chat_memory.messages) >= 20:
        if st.button("סיים שיחה"):
            st.session_state.page = "Result"
            st.rerun()

# Page Result

def page_result():
    st.title("סיכום השיחה")
    
    try:
        with st.spinner("מכין משוב..."):
            # Extract all messages for context
            all_messages = []
            for msg in st.session_state.memory.chat_memory.messages:
                role = "Student" if isinstance(msg, HumanMessage) else "Patient"
                all_messages.append(f"{role}: {msg.content}")
            
            conversation_text = "\n".join(all_messages)
            
            # Create prompt that clearly separates the conversation from instructions
            evaluation_prompt = f"""
            להלן שיחה בין סטודנט לסיעוד ומטופל וירטואלי:

            {conversation_text}

            ---

           
            בהתבסס על השיחה לעיל בלבד, כתוב משוב אישי וישיר לסטודנט/ית בגוף ראשון. התייחס לארבעה מרכיבים עיקריים לפי הסדר הבא:
             1. **זיהוי הבעיה**
            "״פנתה אליך אישה בת 32, שבועיים לאחר ניתוח קיסרי, עם סימני זיהום בפצע הניתוחי."

            2. **אומדן**

                   בדיקות קריטיות: פרט אילו בדיקות נערכו ואילו לא (כגון אומדן פצע ניתוחי ומדידת חום).
                        למשל, 
            ״שאלת את אנה לגבי מצב הפצע הניתוחי, אך לא התייחסת לחום. זו בדיקה קריטית לאומדן."
            
             3. **התמודדות והחלטה**
            התמודדת עם התנגדות המטופלת לפינוי ומתן הסבר לדחיפות המצב.
            ״למשל, זיהית כי אנה חוששת לפנות למיון, אך בעזרת מתן הסבר סייעת לה לקבל החלטה נכונה״
             4. **נקודות לשיפור**
            תן לפחות 2 המלצות ברורות לשיפור            

            """
            
            # Use direct LLM call instead of summarize chain for more control
            feedback = st.session_state.llm.invoke(evaluation_prompt).content
            
            # Display feedback
            st.write(feedback)
            
            # Save feedback to database
            save_result(feedback, datetime.now(), st.session_state.user_email, st.session_state.session_id)


 # הוספת קישור אחרי הצגת המשוב
            st.markdown("""
            ---
            📌 [תודה לכם על ההשתפות בסימולציה, הקליקו פה כדי לענות על שאלון](https://telavivmedicine.fra1.qualtrics.com/jfe/form/SV_cV1yfs9KIQDEEh8)
            """, unsafe_allow_html=True)

    
            # Option to restart
           # if st.button("התחל סימולציה חדשה"):
           #     st.session_state.clear()
              #  st.rerun()
    except Exception as e:
        st.error(f"שגיאה בהכנת המשוב: {e}")
        if st.button("נסה שוב"):
            st.rerun()
# Page Routing
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Chat":
    page_chat()
elif st.session_state.page == "Result":
    page_result()
