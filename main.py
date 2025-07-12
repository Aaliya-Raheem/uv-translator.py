from dotenv import load_dotenv
import os

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load your .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Loaded GEMINI_API_KEY:", "✅" if gemini_api_key else "❌")

# Ensure the key exists
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please define it in your .env file.")

# Set up Gemini as an OpenAI‑compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define the chat model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Configure your run
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Create the agent
agent = Agent(
    name="Translator",
    instructions="You are a helpful translator. Always translate English sentences into clear and simple german."
)

# Run synchronously
response = Runner.run_sync(
    agent,
    input=(
        "My name is Aaliya Raheem. I am a student of Chemistry. "
        "I am very hardworking. I am a QC officer at pharmaceutical indutry."
    ),
    run_config=config
)

# Print only the final translated text
print("→ Translation:\n", response.final_output)
