from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
import os

# ===============================
# 1️⃣ Cargar API Key
# ===============================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ===============================
# 2️⃣ Crear modelo
# ===============================
modelo = ChatOpenAI(model="gpt-5-nano", temperature=0.3)

# ===============================
# 3️⃣ Crear memoria de resumen
# ===============================
memoria = ConversationSummaryMemory(
    llm=modelo,
    memory_key="historial",
    return_messages=False
)

# ===============================
# 4️⃣ Crear prompt y parser
# ===============================
prompt = ChatPromptTemplate.from_template("""
Eres un experto en inversiones en bolsa de valores.
Adapta tus respuestas al nivel de conocimiento financiero del usuario: {nivel_conocimiento}.
Explica de forma clara, educativa y con ejemplos reales cuando sea posible.
Responde siempre en un máximo de 5 oraciones, priorizando claridad y brevedad.

Usuario: {input}
""")

parser = StrOutputParser()

#----pipeline de cadena--------
cadena = prompt | modelo | parser

# ===============================
# 5️⃣ Historial por sesión
# ===============================

#-----------------------ChatMessageHistory------------------------
#Ejemplo de como se almacena la historia en el diccionario store por cada sesion gracias a la funcion obtener_historial con ChatMessageHistory
# store={'perfil_1': ChatMessageHistory(
#     messages=[
#         HumanMessage(content='Quiero invertir en la bolsa de valores.'),
#         AIMessage(content='Perfecto. ¿Tienes un perfil de riesgo conservador o agresivo?'),
#         HumanMessage(content='Soy conservador.'),
#         AIMessage(content='Entonces te convendrían bonos o fondos indexados de bajo riesgo.')
#     ]
# )}
store = {}  # para guardar historiales por sesión

def obtener_historial(session_id: str) -> ChatMessageHistory:
    """Retorna o crea el historial para cada sesión"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



#--------RunnableWithMessageHistory--------
# 1. Obtiene historial desde store     
# 2. Inyecta contexto pasado  
# 3. Ejecuta pipeline prompt→modelo→parser
# 4. Devuelve respuesta   
agente = RunnableWithMessageHistory(
    runnable=cadena,
    get_session_history=lambda session_id: obtener_historial(session_id),
    input_messages_key="input",
)

# ===============================
# 6️⃣ Loop interactivo
# ===============================
if __name__ == "__main__":
    print("🧠 Asistente bursátil con memoria actualizado (LangChain 2025). Escribe 'salir' para terminar.\n")

    # 🔹 Paso adicional: pedir nivel de conocimiento
    nivel_conocimiento = input("📊 Indica tu nivel de conocimiento financiero (principiante / intermedio / avanzado): ").strip().lower()
    session_id = f"usuario_{nivel_conocimiento}"

    print(f"Perfecto. Ajustaré las explicaciones a tu nivel: {nivel_conocimiento}.\n")

    while True:
        pregunta = input("💬 Tu pregunta: ")
        if pregunta.lower().strip() == "salir":
            print("👋 ¡Hasta luego! Recuerda diversificar tus inversiones.")
            break

        # Invocar el agente con el nivel incluido
        respuesta = agente.invoke(
            {"input": pregunta, "nivel_conocimiento": nivel_conocimiento},
            config={"configurable": {"session_id": session_id}}
        )

        # Actualizar resumen de memoria (mantiene contexto condensado)
        try:
            memoria.save_context({"input": pregunta}, {"output": str(respuesta)})
        except Exception as e:
            print(f"[warning] no se pudo actualizar la memoria: {e}")

        print("\n📈 Asistente:", respuesta, "\n")

# siguientes pasos
#------------------1.usar git y github
#------------------2.validador de parametros de entrada
#------------------1.agrega un parametro mas de entrada (ejm:capital, aporte mensual)
#------------------2.carga documentos