import asyncio
import aio_pika

from asyncio import CancelledError

from config import BrokerConfig

from broke.consumer import BrokerConsumer

async def main():
    print("START LLM MODEL")

    try:
        BrokerConfig.CONNECTION = await aio_pika.connect_robust(host=BrokerConfig.RABBITMQ_HOST,
                                                                port=int(BrokerConfig.RABBITMQ_PORT),
                                                                login=BrokerConfig.RABBITMQ_USERNAME,
                                                                password=BrokerConfig.RABBITMQ_PASSWORD,)
    except Exception as e:
        print("CONNECTION ERROR", e)

    try:
        task1 = asyncio.create_task(BrokerConsumer.consume_running_docs_queue())
        task2 = asyncio.create_task(BrokerConsumer.consume_running_query_queue())

        await asyncio.gather(task1, task2)
    except (KeyboardInterrupt, CancelledError):
        return

    print("END LLM MODEL")


if __name__ == '__main__':
    asyncio.run(main())
