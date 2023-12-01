from use_cases.bot import Bot

agent = Bot().agent
while True:
    print("=> Question :")
    question = input()
    print("=> Réponse :")
    print(agent.run(question))
    print("\n")
