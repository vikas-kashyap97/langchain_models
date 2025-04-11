from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv


load_dotenv()
model = ChatAnthropic(model='claude-3.5-sonnet-20241022')

result = model.invoke('What ise 2+2 ?')

print(result.content)