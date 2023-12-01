from use_cases.bot import Bot
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.scoring.eval_chain import LabeledScoreStringEvalChain
from langchain.evaluation.scoring.eval_chain import LabeledScoreStringEvalChain
from langchain.agents import AgentExecutor

def test(agent: AgentExecutor, question: str, reference: str) -> None:
    answer: str = agent.invoke({"input": question})["output"]
    evaluator: LabeledScoreStringEvalChain = LabeledScoreStringEvalChain.from_llm(llm=ChatOpenAI(model="gpt-4"))
    evaluation = evaluator.evaluate_strings(
        prediction=answer,
        reference=reference,
        input=question
    )
    print("=> Question :", question, "\n")
    print("=> Réponse :", answer, "\n")
    print("=> Score :", str(evaluation["score"])  + "/10\n\n")


agent = Bot().agent

# TEST : RAG
print("##### TEST 1 : RAG #####")
question: str = "Donne moi les axes stratégiques du pui bordeaux ?"
reference: str = """Il y a 3 axes stratégiques 
• Axe 1 - Tiss’Innovation : un pacte pour travailler mieux ensemble. 
L’impact sera une optimisation des ressources, un gain de temps et 
une plus grande satisfaction des parties prenantes ;  
• Axe 2 - Declic’Innovation pour rendre l’activité d’innovation 
« attractive ». L’impact sera d’accroitre les opportunités de recherche 
partenariale et les activités de pré-maturation et maturation ;   
• Axe 3 – Impuls’Innovation : pour susciter et accompagner toujours plus 
d’histoires d’innovation. L’impact sera d’accroitre le taux de succès 
d’innovation quelle que soit leur typologie."""
test(agent, question, reference)

# TEST : no tools
print("##### TEST 2 : no tools #####")
question: str = "What is 2+2?"
reference: str = """2 + 2 equals 4."""
test(agent, question, reference)

# TEST 3: memory
print("##### TEST 3 : memory #####")
question: str = "What is my last question?"
reference: str = """Your last question is "What is 2+2?"."""
test(agent, question, reference)

# TEST 4: web search
print("##### TEST 4 : web search #####")
question: str = "A quelle date a eu lieu la perquisition de 2023 chez bpifrance?"
reference: str = """La perquisition des locaux de Bpifrance a eu lieu le 26 juillet 2023 par les enquêteurs de l'Office anti-corruption (Oclciff)."""
test(agent, question, reference)

# TEST 5: wikipedia
print("##### TEST 5 : wikipedia #####")
question: str = "What is a typhoon?"
reference: str = """A typhoon is a tropical cyclone that develops between 180° and 100°E in the Northern Hemisphere.
This region is referred to as the Northwestern Pacific Basin, accounting for almost one-third of the world's annual
tropical cyclones. They can be very destructive for the impacted areas."""
test(agent, question, reference)
