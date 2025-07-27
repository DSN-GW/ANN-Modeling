from sonnet_agent import SonnetAgent


def main():

    agent = SonnetAgent()

    agent.set_system_prompt("You are a helpful AI assistant that provides clear and concise answers.")
    agent.set_parameters(max_tokens=500, temperature=0.3)

    question = "What are the key benefits of using artificial intelligence in healthcare?"
    print(f"Question: {question}")
    print("Answer:")
    response = agent.ask(question)
    print(response)




if __name__ == "__main__":
    main()