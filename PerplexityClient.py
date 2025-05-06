import os
import re
from openai import (
    OpenAI,
    OpenAIError,
)  # Import OpenAIError for more specific exception handling
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()


class PerplexityClient:
    """
    A client for interacting with the Perplexity API using the OpenAI client library.

    Handles API key management, client initialization, and sending requests
    to Perplexity models.
    """

    DEFAULT_BASE_URL = "https://api.perplexity.ai"
    DEFAULT_MODEL = "sonar"
    DEFAULT_SYSTEM_PROMPT = "Be precise and concise."  # Default system prompt

    def __init__(
        self, api_key: str = None, base_url: str = None, default_model: str = None
    ):
        """
        Initializes the PerplexityClient.

        Args:
            api_key (str, optional): Perplexity API key. If None, attempts to load
                                     from the 'PERPLEXITY_API_KEY' environment variable.
            base_url (str, optional): The base URL for the Perplexity API.
                                      Defaults to 'https://api.perplexity.ai'.
            default_model (str, optional): The default model to use for requests.
                                           Defaults to 'sonar'.

        Raises:
            ValueError: If the API key is not provided and cannot be found in
                        environment variables.
        """
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Perplexity API key not provided or found in environment variables (PERPLEXITY_API_KEY)."
            )

        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.default_model = default_model or self.DEFAULT_MODEL

        # Initialize the underlying OpenAI client configured for Perplexity
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            # Catch potential issues during client initialization
            print(f"Error initializing OpenAI client for Perplexity: {e}")
            raise  # Re-raise the exception to halt execution if client can't be created

    def chat(
        self,
        message: str,
        model: str = None,
        system_prompt: str = None,
        messages: list = None,
        stream: bool = False,
    ):
        """
        Sends a message or a list of messages to the Perplexity API chat completions endpoint.

        Args:
            message (str): The user's message (used if messages is None).
            model (str, optional): The Perplexity model to use. Defaults to the client's default_model ('sonar').
            system_prompt (str, optional): A system message to guide the model's behavior. Defaults to "Be precise and concise.".
            messages (list, optional): Full message history as a list of dicts (role/content). If provided, overrides message/system_prompt.
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            str or generator: The content of the assistant's response, or a generator if streaming.
        """
        selected_model = model or self.default_model
        current_system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if messages is not None:
            chat_messages = messages
        else:
            chat_messages = [
                {"role": "system", "content": current_system_prompt},
                {"role": "user", "content": message},
            ]

        try:
            if stream:
                # If the OpenAI client supports streaming, use it
                response = self.client.chat.completions.create(
                    model=selected_model, messages=chat_messages, stream=True
                )
                for chunk in response:
                    if (
                        hasattr(chunk, "choices")
                        and chunk.choices
                        and hasattr(chunk.choices[0], "delta")
                    ):
                        content = getattr(chunk.choices[0].delta, "content", None)
                        if content:
                            yield content
            else:
                response = self.client.chat.completions.create(
                    model=selected_model, messages=chat_messages
                )
                print(response)
                return response.choices[0].message.content

        except OpenAIError as e:
            print(f"Perplexity API Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise


# Example Usage
if __name__ == "__main__":
    try:
        # Instantiate the client
        # The API key is automatically loaded from the environment variable
        perplexity_client = PerplexityClient()

        # Example 1: Using default model (sonar)
        user_input_1 = "How many stars are there in our galaxy?"
        print(f"Sending message (default model): '{user_input_1}'")
        response_1 = perplexity_client.chat(user_input_1)
        print(f"Perplexity response: {response_1}\n")

    except ValueError as e:
        # Catch the error if the API key is missing
        print(f"Configuration Error: {e}")
    except OpenAIError as e:
        # Catch API errors during the chat call
        print(f"API Communication Error: {e}")
    except Exception as e:
        # Catch any other unexpected errors during setup or execution
        print(f"An unexpected error occurred in the main execution block: {e}")
