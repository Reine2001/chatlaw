import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import json
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Charger les variables d'environnement
load_dotenv()

# URL ngrok ou Elasticsearch local
ngrok_url = st.secrets["NGROK_URL"]

# Configuration du modèle Gemini Flash 1.5
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
gen_config = {
    "temperature": 0.5,
    "max_output_tokens": 512
}

gemini_model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config=gen_config
)

# Charger le tokenizer et le modèle DistilBERT pour générer des embeddings
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distil_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Mémoire conversationnelle
memory = ConversationBufferMemory(input_key="question", memory_key="history")

# Prompt LangChain
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Contexte : {context}\nQuestion : {question}\nGénère une réponse en fonction du contexte et de la question ci-dessus."""
)

# Chaîne LangChain avec mémoire de conversation
llm_chain = LLMChain(prompt=prompt, llm=OpenAI(api_key=st.secrets["OPENAI_API_KEY"]), memory=memory)

# Fonction pour créer un embedding à partir d'une question
def generate_embedding(question):
    inputs = tokenizer(question, return_tensors='tf', truncation=True, max_length=512)
    outputs = distil_model(**inputs)
    embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy().flatten()
    return embedding.tolist()

# Fonction pour indexer une nouvelle question et réponse dans Elasticsearch
def index_question_in_elasticsearch(question, response, ngrok_url):
    embedding = generate_embedding(question)
    
    doc = {
        'question': question,
        'response': response,
        'embedding': embedding
    }

    try:
        response = requests.post(
            f'{ngrok_url}/questions_reponses/_doc/', 
            headers={"Content-Type": "application/json"}, 
            data=json.dumps(doc)
        )
        if response.status_code in [200, 201]:
            print("Question, réponse et embedding indexés avec succès.")
        else:
            print(f"Erreur d'indexation : {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête à Elasticsearch : {e}")
# Fonction pour rechercher des articles pertinents dans Elasticsearch
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
            results = response.json()
            return results['hits']['hits']
        else:
            st.error(f"Erreur lors de la recherche : {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion : {e}")
        return None

# Définir la fonction de classification de question selon les institutions
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

# Fonction pour générer une réponse unique en fonction de plusieurs articles
def generate_response_single(question, articles):
    context = "\n\n".join([f"Article {i+1}: {article['text']}" for i, article in enumerate(articles)])
    prompt = f"""Contexte : {context}\nQuestion : {question}\nGénère une réponse pertinente en fonction du contexte ci-dessus et de la question posée en citant tous les articles utilisés dans la réponse."""
    response = gemini_model.generate_content(prompt)
    return response.text

# Fonction pour générer une réponse lorsque la question est classée dans "general_embeddings"
def generate_response_general(question):
    prompt = f"""Question : {question}\nGénère une réponse pertinente à cette question. Notez que cette réponse est générée par un modèle automatique et il est conseillé de la vérifier."""
    response = gemini_model.generate_content(prompt)
    return response.text
# Fonction principale Streamlit
def main():
    st.set_page_config(page_title="Chatbot de réponses juridiques", page_icon="⚖️", layout="wide")

    # Ajout de styles CSS personnalisés avec couleur de fond
    st.markdown("""
        <style>
        /* Appliquer la couleur de fond à l'élément principal de l'application */
        .main {
            background-color: #f0f0f0;
        }

        h1 {
            color: #4CAF50;
            font-size: 3em;
            text-align: center;
        }

        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
        }

        .stTextInput > label {
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)

    # Diviser la page en 3 colonnes principales
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Diviser la colonne du milieu en 3 sous-colonnes
        subcol1, subcol2, subcol3 = st.columns([1, 2, 1])

        # Afficher l'image au centre de la sous-colonne du milieu
        with subcol2:
            st.image("icone1.png", caption="ChatLaw", width=150)

    # Titre principal sous l'image
    st.markdown("<h1>Comment puis-je vous aider?</h1>", unsafe_allow_html=True)

    # Saisie de la question
    query = st.text_input("", placeholder="Tapez ici...")

    # Bouton pour envoyer la question
    if st.button("Envoyer la question"):
        if query:
            with st.spinner('Classification de la question...'):
                category = classify_question(query)

            if category == "general_embeddings":
                with st.spinner('Génération de la réponse...'):
                    response = generate_response_general(query)
                    st.subheader("Réponse générée par le modèle :")
                    st.markdown(f"<strong>{response}</strong>", unsafe_allow_html=True)
                    st.info("Cette réponse a été générée par un modèle automatique. Veuillez vérifier les informations.")
                    index_question_in_elasticsearch(query, response, ngrok_url)
            else:
                with st.spinner('Recherche des articles pertinents...'):
                    articles = search_full_text(ngrok_url, category, query)

                if articles:
                    st.subheader("Articles trouvés :")
                    for i, article in enumerate(articles):
                        with st.expander(f"Article {i+1}"):
                            st.markdown(article['_source']['text'])

                    with st.spinner('Génération de la réponse...'):
                        response = generate_response_single(query, [article['_source'] for article in articles])
                        st.subheader("Réponse générée :")
                        st.markdown(f"<strong>{response}</strong>", unsafe_allow_html=True)
                        index_question_in_elasticsearch(query, response, ngrok_url)
                else:
                    st.error("Aucun article trouvé.")
        else:
            st.error("Veuillez entrer une question avant de soumettre.")

if __name__ == "__main__":
    main()

