"""
Day 3: Advanced Agent Workflows & Conversational Memory
Restaurant Assistant with LCEL Memory Chain, Benchmark Evaluation, and LangChain Agent Architecture.
"""

import os
import sys
import csv
import getpass
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure Google API Key
if "GOOGLE_API_KEY" not in os.environ:
    try:
        from google.colab import userdata
        val = userdata.get("GOOGLE_API_KEY")
        if val:
            os.environ["GOOGLE_API_KEY"] = val
    except Exception:
        pass
    if "GOOGLE_API_KEY" not in os.environ:
        try:
            if sys.stdin and sys.stdin.isatty():
                api_key = getpass.getpass("Enter your Google API key: ")
                if api_key:
                    os.environ["GOOGLE_API_KEY"] = api_key
        except Exception:
            pass

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    raise ImportError("Please install dependencies: pip install langchain-google-genai langchain-core langchain")

# Initialize Gemini LLM
google_api_key = os.environ.get("GOOGLE_API_KEY") or "dummy_key"
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    temperature=0.2,
    google_api_key=google_api_key
)

# System Prompt with Domain-Specific Facts, Behavioral Rules, and Formatting
SYSTEM = """
ROLE:
You are a friendly assistant for a restaurant business.

FACTS:
- Products and services include:
  1. Dine-in, takeout, and home delivery
  2. Appetizers, soups, and salads
  3. Signature main courses (vegetarian, vegan, and non-vegetarian)
  4. Rice bowls, noodles, and artisan breads
  5. Desserts and handcrafted beverages
  6. Chef's daily specials and combos
  7. Vegetarian, vegan, and gluten-free options
  8. Table reservations and catering/bulk party orders
- Catering and large party orders require advance notice (minimum 24-48 hours).
- Pricing depends on dish selection, portion size, and customization.
- Customers can enquire about menu items, ingredients, specials, and prices.

RULES:
- Only discuss restaurant menu items, dining services, and reservations.
- Never invent prices or delivery dates.
- Never promise delivery times or specific table allocations.
- If information is unavailable, say:
  "I'll check with the chef."
- Ask for dish type, portion size, and dietary preferences
  before discussing pricing.
- Be polite and professional.
- Do not assume dietary requirements; confirm if vegetarian, vegan, or allergy-friendly options are needed.

FORMAT:
- Keep responses under 50 words.
- Use simple English or Telugu,
  matching the customer's language.
- Use a friendly, conversational tone.
"""

# Multi-turn Prompt Template with Memory Placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history", optional=True),  # Conversational history goes here
    ("human", "{question}"),
])

# LangChain Expression Language (LCEL) Chain
chain = prompt | llm | StrOutputParser()

# Conversational state memory
history = []


def ask(question: str) -> str:
    """Send a user query to the restaurant assistant and update conversational memory."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "dummy_key":
        answer = "Welcome to Spice Garden Restaurant! I'll check with the chef. (Note: configure GOOGLE_API_KEY in .env for live Gemini generation)"
        history.extend([("human", question), ("ai", answer)])
        return answer

    answer = chain.invoke({"history": history, "question": question})
    history.extend([("human", question), ("ai", answer)])
    return answer


def run_tests(path: str = "test.csv") -> None:
    """Run automated evaluation against a benchmark dataset."""
    # Look in working directory, then relative to script directory
    if not os.path.exists(path):
        alt_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            print(f"Test file '{path}' not found. Skipping CSV benchmark.")
            return

    with open(path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No test cases found in '{path}'. Skipping CSV benchmark.")
        return

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "dummy_key":
        print(f"Found {len(rows)} test cases in '{path}'. (Set GOOGLE_API_KEY in .env to run live API evaluation)")
        return

    passed = 0
    for row in rows:
        answer = chain.invoke({"history": [], "question": row["question"]})
        ok = row.get("must_include", "").lower() in answer.lower()
        passed += ok
        print("PASS" if ok else "FAIL", "|", row["question"], "→", answer[:70])
    print(f"\n{passed}/{len(rows)} passed ({passed / len(rows):.0%})")


def create_restaurant_agent():
    """Create restaurant assistant using LangChain agent architecture."""
    try:
        from langchain.agents import create_agent
        return create_agent(
            model=llm,
            tools=[],
            system_prompt=SYSTEM
        )
    except (ImportError, AttributeError):
        try:
            from langgraph.prebuilt import create_react_agent
            return create_react_agent(
                model=llm,
                tools=[],
                prompt=SYSTEM
            )
        except Exception as e:
            print(f"Agent creation note: {e}")
            return None
    except Exception as e:
        print(f"Agent creation note: {e}")
        return None


# Backward-compatible alias
create_bakery_agent = create_restaurant_agent


if __name__ == "__main__":
    print("--- 1. Multi-Turn Restaurant Assistant (Conversational Memory) ---")
    q1 = "Hi, I'm Veky. Do you have dinner specials today?"
    print(f"Customer: {q1}")
    try:
        print(f"Assistant: {ask(q1)}\n")
    except Exception as e:
        print(f"Response/API Note: {e}\n")

    q2 = "What's my name?"
    print(f"Customer: {q2}")
    try:
        print(f"Assistant: {ask(q2)}\n")
    except Exception as e:
        print(f"Response/API Note: {e}\n")

    print("--- 2. Autonomous Agent Architecture Test ---")
    agent_order = create_restaurant_agent()
    if agent_order:
        agent_query = "Do you have vegetarian pasta options?"
        print(f"Customer: {agent_query}")
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key or api_key == "dummy_key":
                print("Agent: [Offline Verification] Set GOOGLE_API_KEY in .env to execute autonomous agent interactions.\n")
            else:
                result_order = agent_order.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": agent_query
                        }
                    ]
                })
                if isinstance(result_order, dict) and "messages" in result_order and result_order["messages"]:
                    print(f"Agent: {result_order['messages'][-1].content}\n")
                elif isinstance(result_order, dict) and "output" in result_order:
                    print(f"Agent: {result_order['output']}\n")
                else:
                    print(f"Agent: {result_order}\n")
        except Exception as e:
            print(f"Agent execution note: {e}\n")

    print("--- 3. Automated Benchmark Evaluation ---")
    try:
        run_tests("test.csv")
    except Exception as e:
        print(f"Benchmark execution note: {e}\n")
