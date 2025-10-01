import os

from dotenv import load_dotenv


class BrokerConfig:
    load_dotenv()

    RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
    RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
    RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
    RABBITMQ_PORT = os.getenv("RABBITMQ_PORT")

    EMBED_QUERY_QUEUE = "gtw.embed.request.queue"
    EMBED_DOCS_QUEUE = "gtw.docs.request.queue"

    EMBED_EXCHANGE = "gtw.exchange"

    EMBED_QUERY_ROUTING_KEY = "prs.embed.request.routing.key"
    EMBED_DOCS_ROUTING_KEY = "prs.docs.request.routing.key"

    RABBITMQ_URL = f"amqp://{RABBITMQ_USERNAME}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"
    
    CONNECTION = None
