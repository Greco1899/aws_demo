import gradio as gr
import random
import time
import boto3
import json

with gr.Blocks() as demo:

    # Define components and inputsAprove
    endpoint_name = gr.Textbox(label='Endpoint')
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    prompt = f'''You are an helpful Assistant, called Falcon.
    User:{msg}
    Falcon:'''

    def user(user_message, history):
        return gr.update(value='', interactive=False), history + [[user_message, None]]

    # def bot(history):
    #     bot_message = random.choice(['How are you?', 'I love you', 'I'm very hungry'])
    #     history[-1][1] = ''
    #     for character in bot_message:
    #         history[-1][1] += character
    #         time.sleep(0.05)
    #         yield history

    def bot(history):

        # Define Payload
        payload = {
            'inputs': prompt,
            'parameters': {
                'do_sample': True,
                'top_k': 50,
                'top_p': 0.2,
                'temperature': 0.8,
                'max_new_tokens': 1024,
                'repetition_penalty': 1.03,
                'stop': ['\nUser:','<|endoftext|>','</s>']
            }
        }

        # Check for endpoint and invoke LLM
        if endpoint_name != '':
            # Inference
            response = boto3.client('sagemaker-runtime').invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=json.dumps(payload).encode('utf-8'))
            assistant_reply = json.loads(response['Body'].read().decode())
            bot_message = assistant_reply[0]['generated_text'][len(prompt):]
        else:
            bot_message = 'No Endpoint Entered'

        # Stream output
        history[-1][1] = ''
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    # Start event when user submits message
    response = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)

demo.queue(concurrency_count=3)
demo.launch(server_name='0.0.0.0', server_port=7860)
