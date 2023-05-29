# imports
import streamlit as st
from transformers import pipeline
import boto3
import json

st.header('Welcome!')
st.write('\n')
st.write('\n')

# cache model
@st.cache_resource
def init_summarizer_model():
    summarizer_model = pipeline(task='summarization', model='google/pegasus-xsum')
    return summarizer_model

# define model
with st.spinner('Loading'):
    summarizer_model = init_summarizer_model()

# text input
st.write('Input text for summarization:')
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
user_input = st.text_area(label='Paste your own text or use the sample provided.', value=sample_text)
st.write('\n')
st.write('\n')

# summarize text
summarized_text = summarizer_model(user_input)[0]['summary_text']

st.write('Here is your summarized text:')
st.success(summarized_text)
st.write('\n')
st.write('\n')


st.write('Input SageMaker Model Endpoint to use for summarization:')
# define sagemaker endpoint
endpoint = st.text_input(label='SageMaker Model Endpoint')
runtime = boto3.Session().client('sagemaker-runtime')
payload = json.dumps({"inputs": sample_text}).encode('utf-8')

# summarize text
if endpoint != '':
    try:
        response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/json', Body=payload)
        endpoint_summarized_text = json.loads(response['Body'].read().decode())
        
        st.success(endpoint_summarized_text[0]['summary_text'])
    except:
        st.error('Error with Endpoint!')