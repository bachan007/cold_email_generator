# cold_email_generator
The Cold Email Generator project aims to automate the creation of professional and personalized cold emails for job applications. This tool leverages natural language processing to extract relevant information from resumes and company websites, generating tailored emails that highlight the candidate's qualifications and align with the company's needs.
The Cold Email Generator is built using Python and Streamlit, providing an interactive web interface for users to upload resumes and input company URLs. The application processes these inputs to extract key information and generate a cold email. The project utilizes various modules and techniques from the LangChain community, such as document loaders, text splitters, embeddings, and retrieval chains, to efficiently handle and analyze the provided data.

For running the scripts, you need python 3 and Ollama installed in your local system.
Simply clone the project and create a virtual environment using
> python3 -m venv 'path of environment'

and activate the virtual environment

Install the required libraries using 
> pip install -r requirements.txt

The main file here is cold_email_generator file in the folder email_generation_from_company_info.
change the directory and run the scripts using :
> streamlit run cold_email_generator.py
