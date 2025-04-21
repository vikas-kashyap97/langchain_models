from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """Artificial Intelligence (AI) is transforming the way we live, work, and interact with the world. At its core, AI refers to the simulation of human intelligence in machines that are programmed to think and learn. These systems can analyze data, recognize patterns, and make decisions with minimal human intervention. From virtual assistants like Siri and Alexa to self-driving cars and advanced robotics, AI is becoming deeply embedded in everyday life.

In business, AI enhances productivity through automation, predictive analytics, and customer service chatbots. In healthcare, it aids in diagnostics, drug discovery, and personalized medicine. AI also powers recommendation engines on platforms like Netflix and Amazon, creating personalized user experiences.

Despite its benefits, AI raises ethical concerns such as data privacy, algorithmic bias, and job displacement. As AI technology continues to evolve, it's essential to ensure responsible development and transparent governance. The future of AI holds great promise, but it also requires thoughtful collaboration between technologists, policymakers, and society."""

spiltter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    
)

chunks = spiltter.split_text(text)

print(len(chunks))
print(chunks)