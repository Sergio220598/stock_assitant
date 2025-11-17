from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
import os
from agent_rag import agent_rag
from tools import get_stock_price
from langchain.agents import initialize_agent, Tool, AgentType

# ===============================
# 1 Cargar API Key
# ===============================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ===============================
# 2 Crear modelo 
# ===============================
modelo = ChatOpenAI(
    model="gpt-5",
    temperature=0.3,
    streaming=True  
)

# ===============================
# 3 Crear memoria de resumen
# ===============================
memoria = ConversationSummaryMemory(
    llm=modelo,
    memory_key="historial",
    return_messages=False
)

# ===============================
# 4 Crear prompt y parser
# ===============================
prompt = ChatPromptTemplate.from_template("""
Eres un experto en inversiones con amplia experiencia.
Adapta tus respuestas en base a lo siguiente:
- Nivel de conocimiento financiero del usuario: {nivel_conocimiento}
- Monto de capital inicial: {capital} USD
- Inversión mensual: {inversion_mensual} USD

Reglas:
- Explica siempre de forma clara y breve.
- Da ejemplos reales cuando sea posible.
- Si el usuario es principiante, simplifica. Si es avanzado, profundiza.
- Usa el contexto entregado por RAG en tu respuesta.

Usuario: {input}
""")

parser = StrOutputParser()

# ---- pipeline de cadena --------
cadena = prompt | modelo | parser

# ===============================
# 5 Historial por sesión
# ===============================
store = {}

def obtener_historial(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agente = RunnableWithMessageHistory(
    runnable=cadena,
    get_session_history=lambda session_id: obtener_historial(session_id),
    input_messages_key="input",
)

# ===============================
# 6 Tools del agente
# ===============================
tools = [
    Tool(
        name="ObtenerPrecioAccion",
        func=get_stock_price,
        description="Obtiene el precio actual de una acción en USD. Recibe: nombre de la empresa."
    )
]

agente_tools = initialize_agent(
    tools=tools,
    llm=modelo,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# ===============================
# 7 Loop de interacción con STREAMING
# ===============================
if __name__ == "__main__":
    print("🧠 Bienvenido al Asistente de inversiones. Escribe 'salir' para terminar.\n")

    nivel_conocimiento = input("📊 ¿Tu nivel de conocimiento financiero? (principiante / intermedio / avanzado): ").strip().lower()
    capital = input("💰 ¿Capital inicial en USD?: ").strip().lower()
    inversion_mensual = input("📈 ¿Inversión mensual en USD?: ").strip().lower()

    print(f"""
Perfecto. Ajustaré mis explicaciones según:
✓ Nivel de conocimiento: {nivel_conocimiento}
✓ Capital inicial: ${capital}
✓ Inversión mensual: ${inversion_mensual}

Comencemos. 
""")


    session_id = "id_001"

    while True:
        print("==============================================")
        pregunta = input("💬 Tu pregunta: ")

        if pregunta.lower().strip() == "salir":
            print("👋 ¡Hasta luego! Recuerda siempre diversificar tus inversiones.")
            break

        # ---- RAG ----
        contexto_docs = agent_rag(pregunta)
        pregunta_completa = f"{pregunta}\n\nContexto relevante:\n{contexto_docs}"

        print("\n📈 Asistente:\n")

        # STREAMING
        respuesta_completa = ""

        for chunk in agente.stream(
            {
                "input": pregunta_completa,
                "nivel_conocimiento": nivel_conocimiento,
                "capital": capital,
                "inversion_mensual": inversion_mensual
            },
            config={"configurable": {"session_id": session_id}}
        ):
            print(chunk, end="", flush=True)
            respuesta_completa += chunk

        print("\n\n")

        # Guardar en memoria
        try:
            memoria.save_context({"input": pregunta}, {"output": respuesta_completa})
        except Exception as e:
            print(f"[warning] No se pudo actualizar la memoria: {e}")


#mejorar prompts
#conexion a internet
#eliminar carpeta documentos de github
#langGraph y LangSmith


