import time
from functools import wraps
from typing import Callable, Any, Dict
import logging
import asyncio

logger = logging.getLogger("benchmark")

def benchmark_endpoint(num_runs: int = 1):
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            total_processing_time = 0.0
            total_forward_time = 0.0
            max_processing_time = 0.0
            max_forward_time = 0.0

            results = []

            for run in range(1, num_runs + 1):
                start_processing = time.time()
                processed_data = await func(*args, **kwargs)
                end_processing = time.time()

                elapsed_processing = end_processing - start_processing
                total_processing_time += elapsed_processing
                if elapsed_processing > max_processing_time:
                    max_processing_time = elapsed_processing

                start_forward = time.time()
                model_output = await processed_data['inference'](processed_data['data'])
                end_forward = time.time()

                elapsed_forward = end_forward - start_forward
                total_forward_time += elapsed_forward
                if elapsed_forward > max_forward_time:
                    max_forward_time = elapsed_forward

                result = "Not Broken" if model_output[0] > 0.5 else "Broken"
                results.append(result)

                logger.info(f"Run {run}/{num_runs}: Processing time = {elapsed_processing:.4f} sec, Inference time = {elapsed_forward:.4f} sec, Result = {result}")

            average_processing_time = total_processing_time / num_runs
            average_forward_time = total_forward_time / num_runs

            logger.info(f"Average processing time: {average_processing_time:.4f} sec")
            logger.info(f"Max processing time: {max_processing_time:.4f} sec")
            logger.info(f"Average inference time: {average_forward_time:.4f} sec")
            logger.info(f"Max inference time: {max_forward_time:.4f} sec")
            logger.info(f"Number of runs: {num_runs}")

            return {
                "predictions": results,
                "average_processing_time_sec": average_processing_time,
                "max_processing_time_sec": max_processing_time,
                "average_inference_time_sec": average_forward_time,
                "max_inference_time_sec": max_forward_time,
                "num_runs": num_runs
            }

        return wrapper
    return decorator
