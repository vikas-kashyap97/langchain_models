from pydantic import BaseModel, Field 
from typing import Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# JSON Schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "The sentiment of the review either negative, positive or neutral"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "List of pros mentioned in the review"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "List of cons mentioned in the review"
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reciever"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
)

strutured_model = llm.with_structured_output(json_schema)

result = strutured_model.invoke(
    "I recently got my hands on the new Sony headphones after spending quite some time researching the best options out there, and I can confidently say they’ve surpassed my expectations in nearly every way. From the moment I opened the box, the premium build and sleek, modern design stood out — they felt sturdy yet surprisingly lightweight. It’s clear Sony put a lot of thought into the aesthetics and construction. Wearing them for the first time was a delight. The cushions are soft and breathable, making them super comfortable even during marathon listening sessions. But what truly set these headphones apart was the sound quality. Whether I was diving into the rich layers of a classical symphony, vibing with bass-heavy hip-hop tracks, or catching up on podcasts, the audio clarity was phenomenal — crisp highs, deep lows, and an impressively balanced sound profile that felt immersive and satisfying. One of the most impressive features is the noise cancellation. It’s among the best I’ve experienced — it completely shuts out the hum of city traffic and even the buzz of a busy café, allowing me to focus whether I’m working, commuting, or simply relaxing. It genuinely creates this bubble of silence that’s almost therapeutic. Battery life is solid — I easily get through long sessions without needing to recharge. That said, if I had to nitpick, I do wish the battery stretched a bit further, especially since I use them so frequently throughout the day. It’s not a dealbreaker, but a slight improvement here would make them nearly perfect. All in all, I’m thrilled with this purchase. These Sony headphones strike a near-perfect balance between comfort, sound performance, and design. Whether you’re working from home, traveling, or just escaping into your favorite playlist, these are a fantastic choice. Highly recommended.  Name: Vikas Kashyap"
)

print(result)


