import os
import asyncio
import subprocess
import re

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat, Agent
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.contents import AuthorRole
import re
#from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
#    KernelFunctionSelectionStrategy,
#)
#from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
#from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
#from semantic_kernel.contents.chat_message_content import ChatMessageContent
#from semantic_kernel.contents.utils.author_role import AuthorRole
#from semantic_kernel.kernel import Kernel

# --- Configuration ---
# Get Azure OpenAI credentials from environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT","https://capstoneraj.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "77YGGRRQLDiQBblak1bzusOcbRuxIrBSpQh3oNJXtJ6VVGExLvl0JQQJ99BFACYeBjFXJ3w3AAABACOGAtrb")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
    raise ValueError(
        "Please set the environment variables AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME"
    )

class ApprovalTerminationStrategy(TerminationStrategy):

    def __init__(self):
        super().__init__()
        self._is_approved = False
        self._html_code = ""

    """A strategy for determining when an agent should terminate."""
 
    async def should_agent_terminate(self, agent: Agent, history:list[tuple[AuthorRole, str]]) ->bool:
        """Check if the agent should terminate
        Termination occurs when the user explicitly states "APPROVED".
        Also extracts HTML code if present in the last Product Owner message.
        """
        last_message = history[-1]
        last_message_role = last_message[0]
        last_message_content = last_message[1]
        
        # Check for user approval
        if last_message_role == AuthorRole.USER and "APPROVED" in last_message_content.upper():
            print("\nUser has APPROVED. Initiating termination and code push...\n")
            self._is_approved = True
            return True # Terminate the conversation

        # Extract HTML code if the Product Owner has approved and provided it
        if agent.name == "ProductOwner" and "READY FOR USER APPROVAL" in last_message_content.upper():
            html_match = re.search(r"```html\s*([\s\S]*?)```", last_message_content)
            if html_match:
                self._html_code = html_match.group(1).strip()
                print("\nProduct Owner has indicated readiness for user approval and provided HTML code.\n")

        return False # Continue the conversation

    def is_approved(self) -> bool:
        return self._is_approved

    def get_html_code(self) -> str:
        return self._html_code
      
# --- Main Program ---
async def main():
    # 1. Initialize Kernel
    kernel = Kernel()

    # Add Azure OpenAI Chat Completion service
    kernel.add_service(
        AzureChatCompletion(
            service_id="default",
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
    )

    # 2. Create Agent Personas
    business_analyst_instructions = """You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements."""
    software_engineer_instructions = """You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear."""
    product_owner_instructions = """You are the Product Owner which will review the software engineer's code to ensure all user requirements are completed. You are the guardian of quality, ensuring the final product meets all specifications. IMPORTANT: Verify that the Software Engineer has shared the HTML code using the format ```html [code] ```. This format is required for the code to be saved and pushed to GitHub. Once all client requirements are completed and the code is properly formatted, reply with 'READY FOR USER APPROVAL'. If there are missing features or formatting issues, you will need to send a request back to the SoftwareEngineer or BusinessAnalyst with details of the defect."""

    business_analyst = ChatCompletionAgent(
        name="BusinessAnalyst",
        instructions=business_analyst_instructions,
        kernel=kernel,
    )

    software_engineer = ChatCompletionAgent(
        name="SoftwareEngineer",
        instructions=software_engineer_instructions,
        kernel=kernel,
    )

    product_owner = ChatCompletionAgent(
        name="ProductOwner",
        instructions=product_owner_instructions,
        kernel=kernel,
    )


    # 3. Create AgentGroupChat
    approval_strategy = ApprovalTerminationStrategy()
    
    chat = AgentGroupChat(
        agents=[business_analyst, software_engineer, product_owner],
        termination_strategy=approval_strategy,
    )

    # 4. Implement the conversation loop
    print("Welcome to the Multi-Agent App Development System!")
    print("Please describe the web application you'd like to build (e.g., 'Build a simple calculator app').")
    print("Type 'exit' to quit or 'APPROVED' to approve the current code and push to GitHub.")

    while True:
        # Get user input
        user_input = input("\nUSER: ").strip()

        if user_input.lower() == "exit":
            print("Exiting conversation.")
            break
        
        # Add user message to the chat
        await chat.add_chat_message(user_input)

        print("\n--- Agent Responses ---")
        async for message in chat.invoke():
            print(f"#{message.role} - {message.name or '*'}: '{message.content}'")

            # Check for termination condition from the strategy (User approval)
            if approval_strategy.is_approved():
                html_code = approval_strategy.get_html_code()
                if html_code:
                    print("\n--- Code Extraction & Saving ---")
                    try:
                        os.makedirs("output", exist_ok=True)
                        file_path = os.path.join("output", "index.html")
                        with open(file_path, "w") as f:
                            f.write(html_code)
                        print(f"HTML code saved to {file_path}")

                        print("\n--- Calling Git Push Script ---")
                        # Call the bash script to push to GitHub
                        subprocess.run(["bash", "./push_to_github.sh"], check=True)
                        print("Code pushed to GitHub successfully.")
                    except subprocess.CalledProcessError as e:
                        print(f"Error pushing to GitHub: {e}")
                    except Exception as e:
                        print(f"An error occurred during code saving or push: {e}")
                else:
                    print("No HTML code found to save or push, even though 'APPROVED' was detected.")
                
                # After successful push or error, we break the loop as the conversation is approved.
                break 
        
        if approval_strategy.is_approved():
            break # Exit the main loop after handling approval and push

if __name__ == "__main__":
    asyncio.run(main())

