import streamlit as st
import os
import base64
import requests
import json
from dotenv import load_dotenv
from streamlit_chat import message
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
import google.generativeai as genai


load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
ngrok_url = os.environ["NGROK_URL"]


gen_config = {
    "temperature": 0.5,
    "max_output_tokens": 512
}
gemini_model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config=gen_config
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distil_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


def generate_embedding(question):
    inputs = tokenizer(question, return_tensors='tf', truncation=True, max_length=512)
    outputs = distil_model(**inputs)
    embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy().flatten()
    return embedding.tolist()

def index_question_in_elasticsearch(question, response, ngrok_url):
    embedding = generate_embedding(question)
    doc = {
        'question': question,
        'response': response,
        'embedding': embedding
    }
    try:
        requests.post(
            f'{ngrok_url}/questions_reponses/_doc/',
            headers={"Content-Type": "application/json"},
            data=json.dumps(doc)
        )
    except requests.exceptions.RequestException:
        pass

def search_full_text(ngrok_url, index_name, query_text):
    search_query = {
        "query": {
            "match": {
                "text": query_text
            }
        }
    }
    try:
        response = requests.post(f'{ngrok_url}/{index_name}/_search',
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps(search_query))
        if response.status_code == 200:
            return response.json()['hits']['hits']
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def classify_question(query_text):
    droits_humains_keywords = ["handicap", "indigence", "droits humains", "protection des personnes handicapées", "inclusion sociale", "personnes vulnérables","torture"]
    police_judiciaire_keywords = ["violence", "crimes", "délits", "prévention de la violence", "justice pénale", "enquête judiciaire", "procédure pénale","contraventions","circulation routière","stationnements","categories","responsabilité pénale","mineur"]
    police_nationale_keywords = ["académie de police", "formation policière", "forces de l'ordre", "admission stagiaires", "missions de police", "sécurité publique"]
    securite_keywords = ["gardiennage", "sociétés privées de sécurité", "sécurité nationale", "sécurité privée", "protection des biens", "responsabilité professionnelle"]

    if any(keyword in query_text.lower() for keyword in droits_humains_keywords):
        return "droits_humains_embeddings"
    elif any(keyword in query_text.lower() for keyword in police_judiciaire_keywords):
        return "police_judiciaire_embeddings"
    elif any(keyword in query_text.lower() for keyword in police_nationale_keywords):
        return "police_nationale_embeddings"
    elif any(keyword in query_text.lower() for keyword in securite_keywords):
        return "securite_embeddings"
    else:
        return "general_embeddings"

def generate_response_single(question, articles):
    context = "\n\n".join([f"Article {i+1}: {article['text']}" for i, article in enumerate(articles)])
    prompt = f"""Contexte : {context}\nQuestion : {question}\nGénère une réponse pertinente en fonction du contexte ci-dessus et de la question posée en citant tous les articles utilisés dans la réponse."""
    response = gemini_model.generate_content(prompt)
    return response.text

def generate_response_general(question):
    prompt = f"""Question : {question}\nGénère une réponse pertinente à cette question. Notez que cette réponse est générée par un modèle automatique et il est conseillé de la vérifier."""
    response = gemini_model.generate_content(prompt)
    return response.text

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


st.set_page_config(page_title="Chatbot de réponses juridiques", page_icon="⚖️", layout="centered")

image_base64 = get_image_base64("icone1.png")
st.markdown(
    f"<div style='text-align: center;'><img src='data:image/png;base64,{image_base64}' width='150'></div>",
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center;'>LegiChat - Assistant juridique IA</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "article_results" not in st.session_state:
    st.session_state["article_results"] = []

def handle_input():
    query = st.session_state["user_input"]
    if query.strip():
        st.session_state["messages"].append({"role": "user", "content": query})

        with st.spinner("Traitement en cours..."):
            category = classify_question(query)
            if category == "general_embeddings":
                response = generate_response_general(query)
                st.session_state["messages"].append({"role": "assistant", "content": response})
                index_question_in_elasticsearch(query, response, ngrok_url)
                st.session_state["article_results"] = []
            else:
                articles = search_full_text(ngrok_url, category, query)
                if articles:
                    articles_sources = [a['_source'] for a in articles]
                    response = generate_response_single(query, articles_sources)
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    index_question_in_elasticsearch(query, response, ngrok_url)
                    st.session_state["article_results"] = articles
                else:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": "Aucun article pertinent trouvé pour cette question."
                    })
                    st.session_state["article_results"] = []

    st.session_state["user_input"] = ""

for msg in st.session_state["messages"]:
    message(msg["content"], is_user=(msg["role"] == "user"))

st.write("---")


if st.session_state["article_results"]:
    st.markdown("### 📄 Articles utilisés :")
    for i, article in enumerate(st.session_state["article_results"]):
        with st.expander(f"Article {i+1}"):
            st.markdown(article['_source']['text'])

st.text_input(
    "Pose ta question ici",
    key="user_input",
    placeholder="Pose ta question ici...",
    on_change=handle_input
)
