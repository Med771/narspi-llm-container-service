import asyncio

from asyncio import CancelledError

from broke.consumer import BrokerConsumer

async def main():
    print("START LLM MODEL")

    await BrokerConsumer.consume_running_query_queue()
    await BrokerConsumer.consume_running_docs_queue()

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, CancelledError):
        pass

    print("END LLM MODEL")


if __name__ == '__main__':
    asyncio.run(main())
