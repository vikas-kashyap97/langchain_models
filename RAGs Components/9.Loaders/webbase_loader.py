from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421?pid=COMH64PY76CJKBYU&lid=LSTCOMH64PY76CJKBYUOL7TOK&marketplace=FLIPKART&cmpid=content_computer_22371611432_x_8965229628_gmc_pla&tgi=sem,1,G,11214002,x,,,,,,,c,,,,,,,&&cmpid=content_22371611432_gmc_pla&gad_source=1&gclid=EAIaIQobChMIqN-qjfPWjAMVGqhmAh1GjRE5EAQYASABEgIcP_D_BwE'

loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)