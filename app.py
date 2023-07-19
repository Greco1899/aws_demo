# imports
import streamlit as st
from transformers import pipeline
import boto3
import json
import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
st.set_page_config(layout="wide")

# Introduction
col1, col2 = st.columns(2)
with col1:
    st.markdown('# Mirage - SwiftScan')
    st.markdown('## Instant insights from any report or text')
    st.write('Start to summarize and question your document to get to what you really need!')
with col2:
    st.image('aws_demo/Mirage Logo.png', width=250)
st.write('\n')
st.write('\n')

# # Cache model
# @st.cache_resource
# def init_summarizer_model():
#     summarizer_model = pipeline(task='summarization', model='sshleifer/distilbart-cnn-6-6')
#     return summarizer_model

# # Define model
# with st.spinner('Loading'):
#     summarizer_model = init_summarizer_model()

# Text input
st.write('Input text for summarization and extraction:')
sample_text = '''The MAS was founded in 1971 to oversee various monetary functions associated with banking and finance. 
Before its establishment, monetary functions were performed by government departments and agencies. 
The acronym for its name resembles mas, the word for 'gold' in Malay, Singapore's national language - although the acronym is pronounced with each of its initial alphabets.
As Singapore progressed, an increasingly complex banking and monetary environment required more dynamic and coherent monetary administration. 
Therefore, in 1970, the Parliament of Singapore passed the Monetary Authority of Singapore Act leading to the formation of MAS on 1 January 1971. 
The act gives MAS the authority to regulate all elements of monetary policy, banking, and finance in Singapore.
During the COVID-19 pandemic, MAS brought forward its twice yearly meeting from some time in April to 30 March. 
The MAS decided to ease the Singapore dollar's appreciation rate to zero percent, as well as adjust the policy band downwards, the first such move since the Global Financial Crisis. 
This makes it the first time the MAS had taken these two measures together.
Unlike many central banks around the world, the MAS is not independent from the executive branch of the Singaporean government; chairmen of the MAS were from the same political party of the Government. 
Previous chairmen were also either the incumbent or former Ministers of Finance, or were former Prime Ministers or Deputy Prime Ministers of Singapore.'''
user_input = st.text_area(label='Paste your own text or use the sample provided.', value=sample_text, height=200)
st.write('\n')
st.write('\n')

# # Summarize text
# summarized_text = summarizer_model(user_input)[0]['summary_text']

# st.write('Here is your summarized text:')
# st.success(summarized_text)
# st.write('\n')
# st.write('\n')

st.write('What do you want to find out from the text?')
user_question = st.text_input(label='Enter your own question or use the sample provided.', value='When was MAS established?')
st.markdown('''
    Other sample questions:  
    When was Amazon established?  
    What did MAS do during the pandemic?  
    What did ECB do during the pandemic?
    ''')
st.write('\n')
st.write('\n')

st.write('Select LLM to use for summarization and extraction:')
llm_option = st.radio(label='Large Language Model', options=['Anthropic - Claude 2.0', 'TII - Falcon 40B'])
st.write('\n')
st.write('\n')

if llm_option == 'Anthropic - Claude 2.0':
    anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # anthropic = Anthropic(api_key='')

    summarization_prompt='Summarize the following text as a short paragraph:'
    with st.spinner('Generation summary...'):
        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=300,
            prompt=f"{HUMAN_PROMPT} {summarization_prompt+sample_text} {AI_PROMPT}",
        )    
    st.write('LLM Summary:')
    st.success(completion.completion)

    extraction_prompt='Given this CONTEXT, answer the following question. If you do not know the answer, just say that you do not know. Do not try to make up an answer.'
    with st.spinner('Generating answer...'):
        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=300,
            prompt=f"{HUMAN_PROMPT} {'CONTEXT:'+sample_text+extraction_prompt+user_question} {AI_PROMPT}",
            )
    st.write('LLM Response:')
    st.success(completion.completion)

elif llm_option == 'TII - Falcon 40B':
    # llm_endpoint = ''
    llm_endpoint = os.environ.get("FALCON_ENDPOINT")

    summarization_prompt='Write summary of the following TEXT. Do not use more than 100 words. Do not regurgitate from TEXT.'
    # define payload
    prompt = f"""You are an Assistant, called Falcon.
    User:{summarization_prompt+"TEXT:"+"'''"+sample_text+"'''"}
    Falcon:"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "top_k": 40,
            "top_p": 0.9,
            "temperature": 0.2,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.03,
            "stop": ["\nUser:","<|endoftext|>","</s>"]
        }
    }

    # Inference
    with st.spinner('Generating summary...'):
        response = boto3.client('sagemaker-runtime').invoke_endpoint(EndpointName=llm_endpoint, ContentType='application/json', Body=json.dumps(payload).encode('utf-8'))
    response = json.loads(response['Body'].read().decode())
    st.write('LLM Summary:')
    st.success(response[0]['generated_text'][len(prompt):])

    extraction_prompt='Given this CONTEXT, answer the following question. If you do not know the answer, just say that you do not know. Do not try to make up an answer.'
    # define payload
    prompt = f"""You are an Assistant, called Falcon.
    User:{'CONTEXT:'+"'''"+sample_text+"'''"+extraction_prompt+user_question}
    Falcon:"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "top_k": 40,
            "top_p": 0.9,
            "temperature": 0.2,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.03,
            "stop": ["\nUser:","<|endoftext|>","</s>"]
        }
    }

    # Inference
    with st.spinner('Generating answer...'):
        response = boto3.client('sagemaker-runtime').invoke_endpoint(EndpointName=llm_endpoint, ContentType='application/json', Body=json.dumps(payload).encode('utf-8'))
    response = json.loads(response['Body'].read().decode())
    st.write('LLM Response:')
    st.success(response[0]['generated_text'][len(prompt):])


# st.write('Input LLM access key to use for summarization:')
# # Define sagemaker endpoint
# endpoint = st.text_input(label='SageMaker Model Endpoint')
# # Define model type

# # summarize text
# if endpoint != '':
#     payload = json.dumps({"inputs": sample_text}).encode('utf-8')
#     response = boto3.client('sagemaker-runtime').invoke_endpoint(EndpointName=endpoint, ContentType='application/json', Body=payload)
#     endpoint_summarized_text = json.loads(response['Body'].read().decode())
#     st.success(endpoint_summarized_text[0]['summary_text'])
# else:
#     st.write('No Endpoint entered.')



# st.write('Input SageMaker Model Endpoint to use for LLM prompting:')
# # define sagemaker endpoint
# llm_endpoint = st.text_input(label='SageMaker LLM Endpoint')
# user_input = st.text_input(label='User Prompt')
                           
# # summarize text
# if llm_endpoint != '' and user_input != '':
#     # define payload
#     prompt = f"""You are an helpful Assistant, called Falcon.
#     User:{user_input}
#     Falcon:"""

#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "do_sample": True,
#             "top_k": 50,
#             "top_p": 0.2,
#             "temperature": 0.8,
#             "max_new_tokens": 1024,
#             "repetition_penalty": 1.03,
#             "stop": ["\nUser:","<|endoftext|>","</s>"]
#         }
#     }
#     # Inference
#     response = boto3.client('sagemaker-runtime').invoke_endpoint(EndpointName=llm_endpoint, ContentType='application/json', Body=json.dumps(payload).encode('utf-8'))
#     assistant_reply = json.loads(response['Body'].read().decode())
#     st.success(assistant_reply[0]['generated_text'][len(prompt):])
# else:
#     st.write('No Endpoint and User Prompt entered.')
