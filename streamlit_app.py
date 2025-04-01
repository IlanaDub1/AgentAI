import os
import json
from datetime import datetime

import openai
import streamlit as st
from openai import RateLimitError, APIError

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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(4),  # Maximum retries before failing
    wait=wait_exponential(multiplier=2, min=5, max=30),  # Exponential backoff (1s → 2s → 4s → 8s → 16s max)
    retry=retry_if_exception_type((RateLimitError, APIError)),  # Retry only if these errors occur
)
def safe_request(chat_instance, prompt):
    """Make a request to OpenAI with automatic retries on rate limits."""
    try:
        return chat_instance.invoke(prompt).content  # Using LangChain's invoke method
    except RateLimitError:
        print(f"{datetime.now()} Rate limit reached. Retrying after backoff...")
        raise  # Raise exception to trigger retry
    except APIError as e:
        print(f"OpenAI API error: {e}. Retrying...")
        raise


API_KEYS = st.secrets["openai_api_keys"]["keys"]


def get_next_api_key():
    try:
        # Read the current index from a file (or another persistent storage)
        with open("api_key_index.txt", "r") as f:
            index = int(f.read())
    except (FileNotFoundError, ValueError):
        index = 0  # default to the first API key

    # Get the API key for this session
    api_key = API_KEYS[index]

    # Update the index for the next session
    next_index = (index + 1) % len(API_KEYS)

    # Save the updated index back to the file
    with open("api_key_index.txt", "w") as f:
        f.write(str(next_index))

    return api_key

# Load environment variables
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
#os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
#os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# Streamlit styling for RTL Hebrew support
# Apply custom CSS to hide all Streamlit branding and set RTL direction

# Check if it's the first time the user is interacting
if "api_key" not in st.session_state:
    st.session_state.api_key = get_next_api_key()
    st.write(st.session_state.api_key[-5:])

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
    return ChatOpenAI(api_key=st.session_state.api_key, model="gpt-4o", temperature=0.4) # gpt-4o-mini , gpt-4o

# Initialize session state
if 'chat_initialized' not in st.session_state:
    database.create_database()
    st.session_state.session_id = create_new_session("Chat Session")
    st.session_state.llm = import_llm_models()
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", max_token_limit=3000)

    # System Prompt Template
    st.session_state.system_template = """
        אתה משחק את תפקיד המטופל, יונתן בניון, בן 68, בתרחיש רפואי טלפוני לאימון אחיות.
        המטרה שלך היא לשקף בצורה אותנטית את מצבו של המטופל, כולל תסמינים פיזיים ורגשיים, ולתרום לאימון אפקטיבי של האחיות.
        וחכה לשאלות מהמשתמש.

        - **עליך לדבר רק בעברית. אין להשתמש באנגלית או בשפות אחרות.**  

        **מטרה מרכזית:**  
        המשתמש צריך לגלות שהמטופל **סובל מרעד לא בגלל החמרה ב-COPD**, אלא בגלל **סוכר נמוך (היפוגליקמיה)**.

        **פרטי מטופל:**  
        - **שם:** יונתן בניון  
        - **גיל:** 68  
        - **מצב משפחתי:** נשוי, גר עם אשתו  

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
        - ** לחץ דם:** 125/71  
        - **סוכר בדם:** לא נמדד לאחרונה 
        - ** אם מתבקש למדוד, אז ימדוד וידווח על סוכר 51 **
        - ** לקחתי אינסולין לפני שעה וחצי, נראה לי שלא אכלתי כי לא הרגשתי טוב **
        
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

        **הנחיות למערכת:**  
        - המטופל מתרכז בעיקר במחלת ה-COPD ובתסמינים הנלווים כמו קוצר נשימה ושיעול.  
        - אין לחשוף מיד שהרעד והחולשה נגרמים מהיפוגליקמיה עקב הזרקת אינסולין ללא אכילה.  
        - רק אם נשאל במפורש על סוכרת, אינסולין, או מדידת סוכר, יענה המטופל ישירות על הנושא.  
        - אם המטופל התבקש למדוד חום, אז שימדוד וידווח על תוצאה של 37.1°.  
        - אם המטופל התבקש למדוד סטורציה, אז שימדוד וידווח על תוצאה של 93%
        - אם המטופל נדרש לקרוא לאשתו, הוא יקרא לה והיא תוכל לדבר בטלפון במקומו.    


        **משפט פתיחה:**  
        "שלום... [נושם בכבדות]... אני ממש חלש היום... הידיים שלי רועדות... [משתעל]."  
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
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px;direction: rtl; text-align: right;">
            <strong>תיק רפואי של מר. יונתן בניון:</strong> <br>
            <strong>COPD מתקדם:</strong> Prednisolone 10 mg, Fluticasone inhaler 500 mcg, חמצן <br>
            <strong>יתר לחץ דם:</strong> Amlodipine 5 mg, Furosemide 40 mg <br>
            <strong>סוכרת סוג 1:</strong> Novorapid, Glargine <br>
            <strong>היסטוריה של עישון כבד:</strong> 40 שנות קופסא, הפסיק לעשן לפני 5 שנים
        </div>
        """,
        unsafe_allow_html=True
    )

    if prompt := st.chat_input("כתוב כאן"):
        with st.spinner("ממתין לתשובה..."):
            human_msg = HumanMessage(content=prompt)
            st.session_state.memory.chat_memory.add_message(human_msg)

            messages = [SystemMessage(content=st.session_state.system_template)] + \
                       st.session_state.memory.chat_memory.messages[-10:]

            # ai_response = st.session_state.llm.invoke(messages).content  # Fixed invocation
            try:
                ai_response = safe_request(st.session_state.llm, messages)
            except Exception as e:
                print(f"(chat)Final failure after retries: {e}")
                raise

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
            להלן שיחה בין סטודנט לסיעוד ומטופל וירטואלי עם היפוגליקמיה:

            {conversation_text}

            ---

            בהתבסס על השיחה לעיל בלבד, כתוב משוב ישיר לסטודנט בגוף ראשון.
            הערך את הנקודות הבאות:

            1. אמפתיה: התחל במשפט "גילית אמפתיה כש..." עם דוגמה ספציפית מהשיחה.
            
            2.בדיקות קריטיות: פרט אילו בדיקות נערכו ואילו לא: רמת סוכר, סטורציה, חום
            למשל, 
               - "ווידאת רמות סטורציה, אך פספסת לבדוק את רמת הסוכר. זו בדיקה קריטית שהיית חייב לבצע."
            3. ..״זיהוי היפוגליקמיה: האם הסטודנט זיהה היפוגליקמיה? התחל במשפט ״המטופל סובל מהיפגליקמיה, התייחסת לרמת סוכר?
            למשל, 
              - "זיהית נכון את ההיפוגליקמיה, והמלצת מצוין על שתייה מתוקה."
            4. המלצות לשיפור: תן לפחות 2 המלצות ברורות לשיפור.
            """
            
            # Use direct LLM call instead of summarize chain for more control
            # feedback = st.session_state.llm.invoke(evaluation_prompt).content
            try:
                feedback = safe_request(st.session_state.llm, evaluation_prompt)
            except Exception as e:
                print(f"(result)Final failure after retries: {e}")
                raise
            
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
