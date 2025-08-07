import google.generativeai as genai
from config import Settings

settings = Settings()

class LLMService:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name="gemini-2.5-pro")
    def generate_response(self, query: str, search_results: list[str]):
        # context from web search
        context_text = "\n\n".join([
            f"Source {i+1} ({result["url"]}):\n{result["content"]}"
            for i, result in enumerate(search_results)
        ])
        full_prompt = f"""
            Context from web search:
            {context_text}
            query: {query}
            Please provide a comprehensive, detailed, well-cited accurate response using the above context. Think and reason deeply. 
            Ensure it answers the query the user is asking. Do not use your knowledge ultil it is absolutely necessary.
        """

        response = self.model.generate_content(full_prompt, stream=True)
        for chunk in response:
            # yield is useful to get back into the function
            yield chunk.text