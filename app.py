import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub



load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

## Image to text 

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()[0]['generated_text'].capitalize() 


# outputi = query("photo.png")
# print(outputi)

## Text to Story Generation

def generate_story(outputi):
    title_template = PromptTemplate(
        input_variables=["outputi"],
        template="You are a story teller. You can generate a short {outputi} story based on a simple narrative, the story should be no more than 150 words."
    )

    repo_id = "tiiuae/falcon-7b-instruct"

    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.8, "max_new_tokens": 2000})
    hub_chain = LLMChain(prompt=title_template, llm=llm, verbose=True)

    story = hub_chain.run(outputi)
    return story

# story = generate_story(outputi)
# print(story)

## TextStory to Speech Generation

def text2speech(message):
    if os.path.exists("audio.mp3"):
        os.remove("audio.mp3")


    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    payloads = {
        "inputs": message
    }
    
    response = requests.post(API_URL, headers=headers, json=payloads)

    with open('audio.mp3', 'wb') as file:
        file.write(response.content)

# text2speech(story)

## Deployment

def main():
    st.title("Image Captioning with Text-to-Speech")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.read())

    
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        output_text = query("temp_image.png")

        st.subheader("Generated Text:")
        st.write(output_text)

        # Remove the temporary image file
        os.remove("temp_image.png")

        story_text = generate_story(output_text)

        st.subheader("Generated Story:")
        st.write(story_text)


        st.subheader ("Play Audio:")
        text2speech(story_text)
        st.audio('audio.mp3', format='audio/mp3')

if __name__ == "__main__":
    main()
