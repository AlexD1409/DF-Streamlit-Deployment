import streamlit as st
from utility import vector_search, basic_llm_response
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from safety import saftey_checker, verify_scores, extract_summary_scores

load_dotenv()

search_endpoint = os.getenv('search_endpoint')
search_key = os.getenv('search_key')
my_api_key = os.getenv("my_api_key")
search_endpoint = os.getenv("search_endpoint")
embedding_endpoint = os.getenv("embedding_endpoint")
chat_endpoint = os.getenv("chat_endpoint")
my_search_key = os.getenv("my_search_key")

search_index_name = 'dandori-wiki'

keyword_client = SearchClient(endpoint=search_endpoint,
                             index_name=search_index_name,
                             credential=AzureKeyCredential(search_key))

if not all([my_api_key, search_endpoint, embedding_endpoint, chat_endpoint, my_search_key]):
    st.error("Missing one or more environment variables. Please check your .env file.")
    st.stop()


# credential = AzureKeyCredential(my_search_key)
# vector_client = SearchClient(endpoint=search_endpoint, index_name="whimsical-courses-index-vector", credential=credential)
# embedding_client = AzureOpenAI(api_key=my_api_key, api_version="2024-12-01-preview", azure_endpoint=embedding_endpoint)
chat_client = AzureOpenAI(api_key=my_api_key, api_version="2024-12-01-preview", azure_endpoint=chat_endpoint)

st.title('🐼 Dandori PandAI')
my_button=st.button('Click for a suprise!')
if my_button:
    st.balloons()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({
        'role': 'ai',
        'content': "👋 Hi there! I'm your assistant for The School of Dandori. Ask me anything!"
    })


for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


user_input = st.chat_input("Ask something about The School of Dandori...")

if user_input:
    get_vals= saftey_checker(user_input)
    if get_vals: 
        vals = extract_summary_scores(get_vals)
        exceeded = verify_scores(vals)
        # print(vals)
        # print(exceeded)
        if exceeded is None:
            
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            st.chat_message("user").markdown(user_input)

            
            with st.chat_message("ai"):
                with st.spinner("Thinking... 🧠"):
                    try:
                        information_list = []
                        print("Searching for keywords.........................")
                        results = keyword_client.search(user_input, top=1) 
                        
                        for result in results:
                            information_list.append(result['text'])

                        constructed_prompt = f"""
                               You are an expert on answering user inputs about the school of Dandori.
                               Use the information provided here, the first items in the list are the most relevant to the question: {information_list}
                               to answer the following user question: {user_input}
                        """
                        llm_response = basic_llm_response(constructed_prompt, chat_client)
                    
                    except Exception as e:
                        llm_response = f"An error occurred: {str(e)}"
                        st.error(llm_response)

                    st.markdown(llm_response)

                    st.session_state.chat_history.append({'role': 'ai', 'content': llm_response})


        
        else:
            with st.container():
                st.error("🚫 Alert!")
                flagged_metrics = ', '.join(exceeded).replace('_', ' ').title()
                st.markdown(f"**Your input has been flagged for:** {flagged_metrics}")
            
