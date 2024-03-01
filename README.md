# CHONQUE DEL MONTE
![Chonque del Monte](/static/img/LOGO.png)

# Chatbot conversacional utilizando API ChatGPT, Langchaing, Streamlit y Deeplake

Este repositorio contiene un simple pero poderoso chatbot construido con Streamlit, OpenAI, Deep Lake y Langchain. El chatbot mantiene memoria conversacional; puede referenciar eventos e intercambios pasados en sus respuestas.

## Descripción general
Este es un chatbot diseñado para responder preguntas acerca de normativa de Acuicultura,
oficina Castro Servicio Nacional de Pesca y Acuicultura. Por el momento solo tiene conocimientos en su base de datos vectorial acerca de D.S. N° 320 (RAMA). A futuro se incorporarán mas normativas para ampliar su base de conocimiento. Está basado en la API de OpenAI para realizar embeddings de texto a través del uso del último modelo de lenguaje (gpt-3.5-turbo-0125). Utiliza Streamlit como base para la interfaz web, langchain para unificar diferentes plataformas y Deeplake para la memoria en base de datos vectorial. Se optó por simpleza en lugar de implementar muchas características para mantener a futuro de mejor manera los cambios en el código.

### Características principales

- **Streamlit** Framework de Python pusado para crear la interfaz web con el Chatbot
- **GPT OpenAI** Modelo de procesamiento de lenguaje utilizado para generar las respuestas del Chatbot. Adicionalmente se utiliza el modelo text-embedding-3-large para los embeddings.
- **Deep Lake** Plataforma para almacenamiento de datos en una base de datos vectorial
- **LangChain** Librería contenedora para el modelo de ChatGPT que ayuda a manejar el historial de la conversación y estructura las respuestas del modelo.

### Pre requisitos

- Python 3.10
- Streamlit
- Langchain
- Deep Lake API Key 
- OpenAI API Key

## Uso

El Chatbot automáticamente comienza con un sistema de mensaje para dar el tono de la conversación. Luego se ingresan preguntas (idealmente con algo de contexto) para obtener respuestas. Recordar que al consultar acerca de tópicos de carácter legal las respuestas del Chatbot pueden ser erróneas. El historial de conversación se mantiene como contexto para generar futuras respuestas, permitiendo al Chatbot continuidad conversacional. 

## A tener en consideración

Para lograr otorgar mejor contexto al Chatbot, se utilizó una aproximación poco ortodoxa: al mensaje del usuario se agrega otro mensaje (que no aparece desplegado en la ventana de conversación) que le recuerda su rol y los lineamientos a seguir. Esto aumenta considerablemente el número de tokens utilizados, pero al estar diseñado para entregar orientación normativa se prefirió este enfoque a fin de obtener respuestas mas robustas y certeras.
