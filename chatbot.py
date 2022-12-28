import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

st.set_page_config(layout="wide")

def local_css(file_name) :
    with open(file_name) as f : 
        st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)
        
local_css("style.css")

st.header("부산소프트웨어마이스터고 챗봇")

tab1, tab2, tab3 = st.tabs(["학교소개", "입학안내", "문의"])

with tab1 : 
    st.header("저희 소마고를 소개합니다")

with tab2 : 
    st.header("입학 안내")

with tab3 :
    st.header("챗봇에게 무엇이든 물어보세요")

@st.cache(allow_output_mutation=True)
def cached_model() :
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset() : 
    df = pd.read_csv("bsg_chat.csv")
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()


with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 : ', "")
    submitted = st.form_submit_button("전송")

# 응답 response
if 'generated' not in st.session_state : 
    st.session_state['generated'] = []

# 사용자 질문 request
if 'past' not in st.session_state :
    st.session_state['past'] = []

# 없는 질문 예외 처리
if submitted and user_input : 
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x : cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    if answer['distance'] >= 0.5 : 
        st.session_state.generated.append(answer['챗봇'])
    else :
        st.session_state.generated.append("이 사항을 알고 싶으시면 051-971-2153으로 문의주세요")

for i in range(len(st.session_state['past'])) : 
    # message(st.session_state['past'][i], is_user=True, key = str(i) + "_user")
    # if len(st.session_state['generated']) > i :     
    #     message(st.session_state['generated'][i], key = str(i) + "_bot")
    time = datetime.now()
    st.markdown(
    """
        <div class="msg right=msg">
            <div class ="msg-img" style="background-image : url(https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMTEyMzFfNzIg%2FMDAxNjQwOTMyNjEyOTI5.p8_7iNoyyLN7vUaepxn5a_1MXzfvbgcKl0gDVAmRtFYg.OLFaVd2EgB2fcRPpLH790-1dvRc60OfXtDjYruZS8j4g.PNG.moonskinz%2F%25B9%25AE%25B5%25F0%25C0%25DA%25C0%25CE_%25B5%25F0%25BD%25BA%25C4%25DA%25B5%25E5_%25288%2529.png&type=sc960_832)"></div>
            <div class="msg-bubble">
                <div class ="msg-info">
                    <div class = "msg-info-time">{0}:{1}</div>
                </div>
                <p>{2}</p>
            </div>
        </div>
        <div class = "msg left-msg">
            <div class ="msg-img" style="background-image : url(https://search.pstatic.net/sunny/?src=https%3A%2F%2Favatars.githubusercontent.com%2Fu%2F47351101%3Fv%3D4%3Fs%3D400&type=sc960_832)"></div>
            <div class = "msg-bubble">
                <div class = "msg-info">
                    <div class = "msg-info-name">소마고 챗봇</div>
                    <div class = "msg-info-time">{3}:{4}</div>
                </div>
                <p>{5}</p>
            </div>
        </div>
    """.format(time.hour, time.minute, st.session_state['past'][i], time.hour, time.minute, st.session_state['generated'][i])
    , unsafe_allow_html=True)





st.sidebar.title("BSSM")
st.sidebar.info(
    """
    [HomePage](https://school.busanedu.net/bssm-h/main.do)
    [Instagram](https://www.instagram.com/bssm.hs/)
    [Facebook](https://www.facebook.com/BusanSoftwareMeisterHighschool)
    """
    
)
st.sidebar.title("Contact")
st.sidebar.info(
    """
    call : 051- 971-2153
    """
)

