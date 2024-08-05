from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
"""
Provide me the accurate answers regarding the context.
<context>
{context}
<context>
Question:{input}
"""
)


model = Ollama(model='llama3.1')

ollama_embeddings = OllamaEmbeddings(
        model='llama3.1',
        model_kwargs={'device':'gpu'}
    )

def response_from_doc(loader,input_prompt):
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    final_doc = text_splitter.split_documents(document)

    vectors = FAISS.from_documents(final_doc,ollama_embeddings)

    retriever  = vectors.as_retriever()

    document_chain = create_stuff_documents_chain(model,prompt)

    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    response = retrieval_chain.invoke({'input':input_prompt})

    return response['answer']



pdf_loader = PyPDFLoader('hashim')
prompt_for_candidate_info = """Please extract the following details from the provided resume:

    1. **Candidate’s Name**: The full name of the candidate.
    2. **Candidate’s Designation**: The current designation of the candidate that would be clearly mentioned in the resume.
    3. **Industry**: The industry or field in which the candidate has worked.
    4. **Key Skills**: A list of most important and primary skills mentioned in the resume and align with the latest industry trends in the designation of the candidate.
    5. **Achievements**: Notable accomplishments or achievements listed in the resume.

    Provide the extracted information in the following format:

    **Candidate’s Name**: [Name]
    **Industry**: [Industry]
    **Key Skills**: [Skill 1], [Skill 2], [Skill 3]
    **Achievements**: [Achievement 1], [Achievement 2]
    **Current Role**: [Role] at [Current Company]

    Make sure to carefully review the resume and provide accurate and complete details.
        """

candidate_info = response_from_doc(pdf_loader,prompt_for_candidate_info)
print(candidate_info)


url = 'https://www.tranzita.com'

web_loader = WebBaseLoader(url)

prompt_for_company_info = """
Please provide detailed information about the company from the website. I am interested in the following aspects:

1. **Company Overview**: A brief summary of the company’s history, mission, and vision.
2. **Key Areas of Exploration**: What are the primary sectors or industries the company is involved in? Any notable projects or research areas?
3. **Products and Services**: What products or services does the company offer?
4. **Company Structure**: Information about the company's organizational structure, key departments, or teams.
5. **Recent Developments**: Any recent news, updates, or developments related to the company.

Please make sure the information is accurate and up-to-date.

"""

company_info = response_from_doc(web_loader,prompt_for_company_info)
print(company_info)


prompt = ChatPromptTemplate.from_messages([
    ('system','''You are an expert in writing professional cold emails for job applications. 
    Your task is to craft a compelling and personalized email based on the provided details.'''),
    ('user','question:{question}')
])

chain = prompt|model|StrOutputParser()

response = chain.invoke({'question':f"""
Please write a cold email for a job application based on the following information:
**Candidate’s Resume Highlights:**
{candidate_info}
**Company Description:**
{company_info}
**Email Objective:**
- The goal of this email is to introduce candidate_name with his designation, highlight their qualifications, their most recent profile overview and express interest in potential job opportunities which match with their recent projects and tech stack at company_name.
**Desired Outcome:**
- Request a meeting or call to discuss potential opportunities.
Make sure the email is professional, concise, and tailored to the company's needs and values.
"""})

print(response)