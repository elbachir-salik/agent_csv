import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()

    # Set page configuration
    st.set_page_config(page_title='Ask Your CSV', page_icon='📊', layout='wide')

    # Add a title and description
    st.title("Ask Your CSV 📊")
    st.markdown("""
        **Upload a CSV file and ask questions about its contents.**
        This app uses a powerful AI to understand and analyze your data.
    """)

    # File uploader for CSV
    st.sidebar.header("Upload Your CSV File")
    user_csv = st.sidebar.file_uploader('Choose a CSV file', type="csv")

    # Instructions and example questions
    st.sidebar.markdown("""
        ### How to use:
        1. Upload your CSV file using the uploader above.
        2. Enter your question in the input box below.
        3. Click 'Submit' to get the answer.
        
        ### Example Questions:
        - What is the total sales for the month of June?
        - How many unique products are there?
        - What is the average price of products?
    """)

    if user_csv is not None:
        st.subheader("Ask a Question About Your CSV Data")
        user_question = st.text_input("What do you want to know from your CSV?")

        if user_question:
            # Display the user's question
            st.write(f'**You asked:** {user_question}')

            # Initialize the AI model
            llm = AzureChatOpenAI(
                azure_deployment="gpt-4",
                openai_api_version="2023-05-15",
                azure_endpoint="https://aibcp4.openai.azure.com/",
                max_tokens=1400,
                api_key="a9b5778f059648b7863c397ff8f8248a",
            )

            # Create the CSV agent
            agent = create_csv_agent(llm=llm, path=user_csv, verbose=True, allow_dangerous_code=True)

            # Get the response from the agent
            with st.spinner('Analyzing your data...'):
                response = agent.run(user_question)

            # Display the response
            st.success("Here is the answer:")
            st.write(response)
        else:
            st.info("Please enter a question to get an answer.")

if __name__ == '__main__':
    main()
