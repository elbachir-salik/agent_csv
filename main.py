import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()
    

    st.set_page_config(page_title= 'ask your csv')    
    st.header("ask your csv")
    user_csv = st.file_uploader('upload your csv file',type="csv")

    if user_csv is not None:
        user_question = st.text_input("what do you want to know from your csv:   ")

        # llm = ChatOpenAI(temperature=0)
        llm = AzureChatOpenAI(
                azure_deployment="gpt-4",
                openai_api_version="2023-05-15",
                azure_endpoint="https://aibcp4.openai.azure.com/",
                max_tokens=1400,
                api_key="a9b5778f059648b7863c397ff8f8248a",
            )
        agent = create_csv_agent(llm= llm, path= user_csv, verbose=True,allow_dangerous_code=True)
    
        if user_question is not None and user_question != "":
            st.write(f'you ask:{user_question}')

            response = agent.run(user_question)
            st.write(response)



if __name__ == '__main__':
    main()