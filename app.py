import os
import json
import pickle
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# =========================================================
# CONFIG & OPENAI
# =========================================================
load_dotenv()
MODEL_NAME = "gpt-4o-mini" # Atualizado para gpt-4o-mini que é a versão recomendada
MAX_HISTORY = 10
MAX_REACT_STEPS = 3

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================================================
# LOAD MODEL & MAPS
# =========================================================
with open("modelo_titanic.pkl", "rb") as f:
    modelo = pickle.load(f)

SEX_MAP = {"masculino": 0, "male": 0, "homem": 0, "feminino": 1, "female": 1, "mulher": 1}
EMBARKED_MAP = {"c": 0, "q": 1, "s": 2}

CONTEXTO_TITANIC = (
    "Mulheres, crianças e passageiros da 1ª classe tiveram prioridade absoluta e alta taxa de sobrevivência. "
    "Homens adultos e passageiros da 3ª classe tiveram baixa prioridade, botes distantes e menor taxa de sobrevivência."
)

# =========================================================
# TOOLS IMPLEMENTATION
# =========================================================
def prever_modelo(sexo, idade, classe, sibsp=0, parch=0, tarifa=33.2, embarque="S"):
    """Executa o modelo de ML para prever a sobrevivência com base em dados demográficos."""
    try:
        sex_val = SEX_MAP.get(str(sexo).lower(), 0)
        emb_val = EMBARKED_MAP.get(str(embarque).lower(), 2)
        
        entrada = pd.DataFrame([{
            "Age": float(idade),
            "Fare": float(tarifa),
            "Sex": sex_val,
            "Pclass": int(classe),
            "sibsp": int(sibsp),
            "Parch": int(parch),
            "Embarked": emb_val,
        }])
        
        prob = modelo.predict_proba(entrada)[0][1] * 100
        resultado = "SOBREVIVERIA" if prob >= 50 else "NÃO SOBREVIVERIA"
        return json.dumps({"resultado": resultado, "probabilidade_sobreviver": f"{prob:.1f}%"})
    except Exception as e:
        return json.dumps({"erro": f"Dados insuficientes ou inválidos: {str(e)}"})

def buscar_contexto(query):
    """Retorna fatos históricos do Titanic para explicar os motivos de sobrevivência."""
    return json.dumps({"contexto_historico": CONTEXTO_TITANIC})

# =========================================================
# DEFINIÇÃO DAS TOOLS (SCHEMA OPENAI)
# =========================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "prever_modelo",
            "description": "Calcula a probabilidade de sobrevivência de um passageiro. Use sempre que o usuário fornecer sexo, idade e classe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sexo": {
                        "type": "string", 
                        "enum": ["masculino", "feminino"], 
                        "description": "Gênero do passageiro. Infira 'masculino' se o usuário disser 'homem' ou 'rapaz', e 'feminino' se disser 'mulher' ou 'moça'."
                    },
                    "idade": {"type": "number", "description": "Idade em anos"},
                    "classe": {"type": "integer", "enum": [1, 2, 3], "description": "Classe do bilhete (1, 2 ou 3)"}
                },
                "required": ["sexo", "idade", "classe"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_contexto",
            "description": "Busca o contexto histórico e motivos do Titanic quando o usuário perguntar 'por que', 'qual o motivo' ou pedir explicações.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "O termo ou pergunta de busca histórica"}
                },
                "required": ["query"]
            }
        }
    }
]

# =========================================================
# SYSTEM PROMPT (Instruções do Agente)
# =========================================================
SYSTEM_PROMPT = """Você é o Agente IA Oficial do Titanic. 

Seu objetivo é responder se passageiros sobreviveriam usando o modelo de ML ou explicar o porquê usando o contexto histórico.

REGRAS E FLUXO:
1. Se faltarem dados essenciais (sexo, idade ou classe) para realizar a previsão, NÃO invente. Responda pedindo educadamente os dados ausentes.
2. Se o usuário perguntar 'Por que?' ou pedir justificativas, use a ferramenta 'buscar_contexto' para fundamentar sua resposta com dados históricos.
3. Responda de forma direta, clara e em no máximo duas frases.
"""

# =========================================================
# LOOP REACT DO AGENTE
# =========================================================
historico = [{"role": "system", "content": SYSTEM_PROMPT}]

def limpar_historico(hist, max_len):
    """Garante que o histórico não seja cortado quebrando a relação tool_calls -> tool"""
    if len(hist) <= max_len:
        return hist
        
    sys_msg = hist[0]
    corte = hist[-(max_len-1):]
    
    # Remove mensagens soltas no início do corte até encontrar a vez do usuário
    while corte and corte[0].get("role") != "user":
        corte.pop(0)
        
    return [sys_msg] + corte

def agente(pergunta_usuario):
    global historico
    
    historico.append({"role": "user", "content": pergunta_usuario})
    historico = limpar_historico(historico, MAX_HISTORY)

    for _ in range(MAX_REACT_STEPS):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=historico,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0
        )
        
        msg = response.choices[0].message
        
        # Se o LLM decidiu que precisa acionar uma ferramenta (Action)
        if msg.tool_calls:
            # exclude_none=True evita o envio de campos nulos que causam erro 400
            historico.append(msg.model_dump(exclude_none=True))
            
            for tool_call in msg.tool_calls:
                nome_funcao = tool_call.function.name
                argumentos = json.loads(tool_call.function.arguments)
                
                # Executa a ferramenta correta
                if nome_funcao == "prever_modelo":
                    resultado_tool = prever_modelo(**argumentos)
                elif nome_funcao == "buscar_contexto":
                    resultado_tool = buscar_contexto(**argumentos)
                else:
                    resultado_tool = "Ferramenta não encontrada."
                
                # Devolve a resposta da ferramenta para o LLM (Observation)
                historico.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": resultado_tool
                })
        else:
            # Resposta Final do LLM (Output final em linguagem natural)
            resposta_final = msg.content.strip() if msg.content else "Não foi possível gerar uma resposta."
            historico.append({"role": "assistant", "content": resposta_final})
            return resposta_final
            
    return "Não consegui processar a requisição dentro do limite do loop ReAct."

# =========================================================
# INTERFACE TERMINAL
# =========================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Olá bem vindo ao 'AGENTE TITANIC' ")
    print("Digite 'sair' para encerrar.")
    print("=" * 50)

    while True:
        pergunta = input("\nPergunta: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando agente.")
            break
            
        try:
            resposta = agente(pergunta)
            print(f"\n→ {resposta}")
        except Exception as e:
            print(f"\nErro no sistema: {e}")