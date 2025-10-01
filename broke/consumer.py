import asyncio
import json

import aio_pika

from asyncio import CancelledError

from helper.refactor import RefactorHelper

from decorator import BrokerDecorator

from config import BrokerConfig

class BrokerConsumer:
    @staticmethod
    @BrokerDecorator.log_call(prefix=f"consume: {BrokerConfig.EMBED_DOCS_QUEUE}")
    async def consume_docs_queue():
        connection = BrokerConfig.CONNECTION

        if not connection:
            exit("CONNECTION ERROR")

        async with connection:
            channel = await connection.channel()

            await channel.set_qos(prefetch_count=10)

            queue = await channel.declare_queue(BrokerConfig.EMBED_DOCS_QUEUE, durable=True)

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        obj = json.loads(message.body.decode())

                        if "uuid" not in obj or "chunks" not in obj:
                            continue

                        print(f"[>] Received from {BrokerConfig.EMBED_DOCS_QUEUE}: UUID={obj['uuid']}")

                        res: dict = RefactorHelper.get_docs_embedding(obj=obj)

                        if message.reply_to:
                            await channel.default_exchange.publish(
                                aio_pika.Message(
                                    body=json.dumps(res).encode(),
                                    correlation_id=message.correlation_id
                                ),
                                routing_key=message.reply_to
                            )

    @staticmethod
    @BrokerDecorator.log_call(prefix=f"consume: {BrokerConfig.EMBED_QUERY_QUEUE}")
    async def consume_query_queue():
        connection = BrokerConfig.CONNECTION

        if not connection:
            exit("CONNECTION ERROR")

        async with connection:
            channel = await connection.channel()

            await channel.set_qos(prefetch_count=10)

            queue = await channel.declare_queue(BrokerConfig.EMBED_QUERY_QUEUE, durable=True)

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        obj = json.loads(message.body.decode())

                        if "uuid" not in obj or "query" not in obj:
                            continue

                        print(f"[>] Received from {BrokerConfig.EMBED_QUERY_QUEUE}: UUID={obj['uuid']}")

                        res: dict = RefactorHelper.get_query_embedding(obj=obj)

                        if message.reply_to:
                            await channel.default_exchange.publish(
                                aio_pika.Message(
                                    body=json.dumps(res).encode(),
                                    correlation_id=message.correlation_id
                                ),
                                routing_key=message.reply_to
                            )

    @staticmethod
    @BrokerDecorator.log_call(prefix="consume_running_query_queue")
    async def consume_running_query_queue():
        try:
            await BrokerConsumer.consume_query_queue()
        except CancelledError as e:
            print("Cancelled Error in consume_running_query_queue: {}".format(e))
        except Exception as e:
            print("Unknown Error in consume_running_query_queue: {}".format(e))

    @staticmethod
    @BrokerDecorator.log_call(prefix="consume_running_docs_queue")
    async def consume_running_docs_queue():
        try:
            await BrokerConsumer.consume_docs_queue()
        except CancelledError as e:
            print("Cancelled Error in consume_running_docs_queue: {}".format(e))
        except Exception as e:
            print("Unknown Error in consume_running_docs_queue: {}".format(e))
