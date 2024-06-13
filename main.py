import streamlit as st
from typing import Generator
from groq import Groq
import json
import requests
from streamlit_lottie import st_lottie

# Configurações iniciais da página
st.set_page_config(page_icon="🤖", layout="wide", page_title="Nexus")

# Função para carregar um arquivo Lottie
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Função para carregar uma animação Lottie a partir de uma URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Carregando animações Lottie
lottie_coding = load_lottiefile("lottiefiles\config.json")
lottie_hello = load_lottieurl("https://lottie.host/2a4cd888-a67b-4535-b6c5-8cc74f789fde/9zIajAtwwn.json")

# Estilizando os componentes de seleção
st.markdown(
    """
    <style>
    .stSlider label, .stSelectbox label {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Criando um layout de colunas para centralizar a animação Lottie
col1, col2, col3, col4, col5 = st.columns([1, 2, 3, 4, 5])

with col4:
    st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",  # medium ; high
        height=150,
        width=250,
        key=None,
    )

# Função para mostrar um ícone de emoji
def icon(emoji: str):
    """Mostra um emoji como ícone de página estilo Notion."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

# Subtítulo da página
st.subheader("Seja Bem-Vindo(a) ao Nexus! :)", divider="blue", anchor=False)

# Inicializando o cliente da API Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Inicializando o histórico de mensagens e o modelo selecionado
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Detalhes dos modelos disponíveis
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Layout para seleção do modelo e ajuste de max_tokens
col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Escolha seu modelo:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=4  # Padrão para mixtral
    )

# Detectar mudança de modelo e limpar histórico de chat se o modelo mudou
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with col2:
    # Ajustar o controle deslizante de max_tokens dinamicamente com base no modelo selecionado
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,  # Valor mínimo para permitir alguma flexibilidade
        max_value=max_tokens_range,
        # Valor padrão ou máximo permitido se menor
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )

# Exibir mensagens do chat a partir do histórico em execução do app
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Função para gerar respostas de chat a partir da resposta da API Groq
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Gera o conteúdo da resposta do chat a partir da resposta da API Groq."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Entrada de texto para o prompt do usuário
if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Obter resposta da API Groq
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Usar a função geradora com st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Adicionar a resposta completa ao session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Lidar com o caso em que full_response não é uma string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})
