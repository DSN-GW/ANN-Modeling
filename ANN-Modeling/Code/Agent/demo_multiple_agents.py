from sonnet_agent import SonnetAgent


def demo_different_agents():
    print("=== Sonnet Agent Class Reusability Demo ===\n")
    
    print("1. Creating a Code Review Agent:")
    code_reviewer = SonnetAgent()
    code_reviewer.set_system_prompt("You are an expert code reviewer. Analyze code for best practices, potential bugs, and improvements.")
    code_reviewer.set_parameters(max_tokens=500, temperature=0.1)
    
    print("2. Creating a Creative Writing Agent:")
    creative_writer = SonnetAgent()
    creative_writer.set_system_prompt("You are a creative writing assistant. Help with storytelling, character development, and narrative structure.")
    creative_writer.set_parameters(max_tokens=800, temperature=0.7)
    
    print("3. Creating a Technical Documentation Agent:")
    tech_writer = SonnetAgent()
    tech_writer.set_system_prompt("You are a technical documentation specialist. Create clear, concise, and well-structured technical documentation.")
    tech_writer.set_parameters(max_tokens=1000, temperature=0.2)
    
    print("4. Creating a Data Analysis Agent:")
    data_analyst = SonnetAgent()
    data_analyst.set_system_prompt("You are a data analysis expert. Help interpret data, suggest analysis methods, and explain statistical concepts.")
    data_analyst.set_parameters(max_tokens=600, temperature=0.3)
    
    agents = {
        "Code Reviewer": code_reviewer,
        "Creative Writer": creative_writer,
        "Technical Writer": tech_writer,
        "Data Analyst": data_analyst
    }
    
    print("All agents created successfully!")
    print("Each agent has different system prompts and parameters.")
    print("They can be used independently for their specific purposes.")
    print("\nExample usage:")
    print("- code_reviewer.ask('Review this Python function for improvements')")
    print("- creative_writer.ask('Help me develop a character for my story')")
    print("- tech_writer.ask('Document this API endpoint')")
    print("- data_analyst.ask('Explain the significance of this correlation')")
    
    return agents


if __name__ == "__main__":
    demo_different_agents()