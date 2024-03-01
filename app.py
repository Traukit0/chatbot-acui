# Imports
import os 
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv 
import streamlit as st 
from streamlit_chat import message

# CSS Adicional para el efecto de fuente y fondo. Esto quizás podría ir en un archivo aparte para limpiar el código
css = """
<style>
    /* style.css */

    body {
        font-family: 'sans-serif';
        color: #262730;
    }

    h1, h2, h3 {
        color: #4E2A84;
    }

    .stButton>button {
        border: 1px solid #4E2A84;
        color: #FFFFFF;
        background-color: #4E2A84;
    }

    .stTextInput>div>div>input {
        border-radius: 20px;
    }

    .stDataFrame {
        border: 1px solid #4E2A84;
    }

</style>
"""
# load_dotenv() # Descomentar para desarrollo local

# Inicializar variables ambientales y APIs
# Cambiar a os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY') para desarrollo local
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['ACTIVELOOP_TOKEN'] = st.secrets['ACTIVELOOP_TOKEN']
os.environ['DEEPLAKE_ACCOUNT_NAME']= st.secrets['DEEPLAKE_ACCOUNT_NAME']


st.set_page_config(page_title='CHATBOT ACUICULTURA', layout = 'centered', page_icon = 'random', initial_sidebar_state = 'auto')
st.title("CHATBOT NORMATIVA ACUICULTURA")
st.markdown(css, unsafe_allow_html=True)


# Inicializar variables de estado
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Estoy listo para responder tus preguntas"]  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hola!"]  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input



@st.cache_data
def doc_preprocessing():
    """ Toma los archivos base y los corta en chunks para proceso """
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # Solamente lee archivos PDF
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embeddings_store():
    """ Utiliza embeddings de OpenAI
        Almacena los archivos procesados en una base de datos vectorial en DeepLake
        Luego devuelve esa misma base para consultas """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print(embeddings)
    texts = doc_preprocessing()
    db = DeepLake.from_documents(
        texts,
        embeddings,
        dataset_path=f"hub://mcanonesbet/chat-norm")
    print(db)
    db = DeepLake(
        dataset_path=f"hub://mcanonesbet/chat-norm",
        read_only=True,
        embedding=embeddings,
    )
    return db

@st.cache_resource
def search_db():
    """ Ejecuta consultas sobre la base de datos vectorial en DeepLake """
    db = embeddings_store()
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    # retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    model = ChatOpenAI(model='gpt-3.5-turbo-0125')
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa

qa = search_db()

# Desplegar historial de conversación usando streamlit_messages
def display_conversation(history):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Mostrar respuesta de IA
        message(st.session_state["generated"][i], key=str(i))
        # Mostrar mensaje de usuario
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')

# Función para volver a dejar en blanco el cuadro de input de texto 
def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ''

def main():
    # Íngresar texto en Streamlit
    st.text_input("Ingresar consulta: ", key="prompt_input", on_change=submit)
    user_input = st.session_state.entered_prompt
        
    # Buscar un query en la database basada en la pregunta + un texto de contexto
    if user_input:
        output = qa({'query': user_input+""", Eres un asistente en materias de normativa acuícola,
                     siempre listo para responder dudas y consultas acerca de la normativa almacenada
                     en la base de datos. Tu misión es entregar información de alta calidad,
                     de la manera más clara posible a las preguntas planteadas, dentro del
                     conocimiento adquirido. No entregar respuestas falsas, en el caso de no tener
                     la respuesta responder "No lo sé". El tono debe siempre formal y cortés,
                     haciendo los mejores esfuerzos para entregar respuestas correctas a las preguntas
                     planteadas """})
        # print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        display_conversation(st.session_state)

    # Créditos
    st.markdown("""
    <style>
    .centered-text {
        text-align: center;
    }
    </style>
    <div class="centered-text">
    Hecho por Manuel Cano N., Castro, 2024
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
