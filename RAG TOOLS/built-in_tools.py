# pip install -qU duckduckgo-search langchain-community

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke("Obama's first name?")

print(result)



# pip install --upgrade --quiet langchain-community

from langchain_community.tools import ShellTool

shell_tool = ShellTool()

result = shell_tool.invoke("ls")

print(result)

