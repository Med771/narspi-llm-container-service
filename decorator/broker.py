import functools
import asyncio

from aio_pika.exceptions import ConnectionClosed, AMQPException

from tools.logger import LoggerHelper

logger = LoggerHelper.get_logger(name="decorator", module="broke", error=True)

class BrokerDecorator:
    @staticmethod
    def log_call(prefix):
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)

                    return result
                except ConnectionClosed as e:
                    logger.error(msg=f"CAll: {prefix} Broker Connection Error: {e}, data: {args}, {kwargs}", exc_info=True)
                except AMQPException as e:
                    logger.error(msg=f"CAll: {prefix} Broker AMQP Error: {e}, data: {args}, {kwargs}")
                except Exception as e:
                    logger.error(msg=f"CAll: {prefix} Broker Exception: {e}", exc_info=True)

                return None

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)

                    return result
                except ConnectionClosed as e:
                    logger.error(msg=f"CAll: {prefix} Broker Connection Error: {e}, data: {args}, {kwargs}", exc_info=True)
                except AMQPException as e:
                    logger.error(msg=f"CAll: {prefix} Broker AMQP Error: {e}, data: {args}, {kwargs}")
                except Exception as e:
                    logger.error(msg=f"CAll: {prefix} Broker Exception: {e}", exc_info=True)

                return None

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator