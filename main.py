from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chatbot")
def chatbot():
    return render_template("med_chatbot.html")


@app.route("/get", methods=["POST"])
def get_response():
    user_question = request.form["msg"]  # Get the user's message from the form data
    response = user_input(user_question)  # Get the response from the bot
    return jsonify({"response": response})  # Send the response back to the client


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_name = "medicalchatbot2"
    # vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace="example-namespace")
    # docs = vectorstore.similarity_search(user_question, namespace="example-namespace")

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    prompt_template = """
    you are a Medical Assistant. If user writes name of the disease, then explain the symptoms and treatment for that disease. Answer the question as detailed as possible, make sure to provide all the details, if the answer is not in provided context then create an answer by yourself that makes sense. give the answer point wise.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    convo = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    "you are a Medical Assistant. If user writes name of the disease, then explain the symptoms and treatment for that disease. Also help user identify the disease based on the symptoms user is having and give him information about treatment. Answer the question as detailed as possible, make sure to provide all the details. give the answer point wise."
                ],
            },
            {
                "role": "model",
                "parts": [
                    "Got it! I'm ready to put on my Medical Assistant hat.  Just tell me the name of the disease you're curious about, or describe the symptoms you're experiencing, and I'll do my best to help you identify the potential cause and provide information about its treatment.  I'll make sure to be as thorough as possible, giving you details point by point."
                ],
            },
        ]
    )

    response = convo.send_message(user_question)
    response_text = convo.last.text  # Extract the text from the ChatSession object
    return response_text


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()

