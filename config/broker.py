import os

from dotenv import load_dotenv


class BrokerConfig:
    load_dotenv()

    RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
    RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
    RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
    RABBITMQ_PORT = os.getenv("RABBITMQ_PORT")

    EMBED_QUERY_QUEUE = "embed.query"
    EMBED_DOCS_QUEUE = "embed.docs"
    EMBED_VECTOR_QUEUE = "embed.vector"

    EMBED_EXCHANGE = "narspi-embed"

    EMBED_QUERY_ROUTING_KEY = "embed.query.routing.key"
    EMBED_DOCS_ROUTING_KEY = "embed.docs.routing.key"
    EMBED_VECTOR_ROUTING_KEY = "embed.vector.router.key"

    RABBITMQ_URL = f"amqp://{RABBITMQ_USERNAME}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"