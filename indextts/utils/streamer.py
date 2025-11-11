from __future__ import annotations

import torch

import asyncio
from queue import Queue
from typing import TYPE_CHECKING, Optional


from transformers.generation import BaseStreamer


class AudioStreamer(BaseStreamer):
    """
    Audio streamer that stores audio chunks in queues for each sample in the batch.
    This allows streaming audio generation for multiple samples simultaneously.
    
    Parameters:
        batch_size (`int`):
            The batch size for generation
        stop_signal (`any`, *optional*):
            The signal to put in the queue when generation ends. Defaults to None.
        timeout (`float`, *optional*):
            The timeout for the audio queue. If `None`, the queue will block indefinitely.
    """
    
    def __init__(
        self, 
        stop_signal: Optional[any] = None,
        timeout: Optional[float] = None,
    ):
        self.stop_signal = stop_signal
        self.timeout = timeout
        
        # Create a queue for each sample in the batch
        self.audio_queues = Queue()
        self.finished_flags = False
        self.sample_indices_map = {}  # Maps from sample index to queue index
        
    def put(self, audio_chunks: torch.Tensor):
        """
        Receives audio chunks and puts them in the appropriate queues.
        
        Args:
            audio_chunks: Tensor  containing audio chunks
        """
        audio_chunk = audio_chunks.detach().cpu()
        self.audio_queues.put(audio_chunk, timeout=self.timeout)
    
    def end(self):
        """
        Signals the end of generation for specified samples or all samples.
        
        Args:
            sample_indices: Optional tensor of sample indices to end. If None, ends all.
        """

        self.audio_queues.put(self.stop_signal, timeout=self.timeout)
        self.finished_flags = True
    
    def __iter__(self):
        """Returns an iterator over the batch of audio streams."""
        return AudioBatchIterator(self)
    
    def get_stream(self):
        """Get the audio stream for a specific sample."""

        return AudioSampleIterator(self)


class AudioSampleIterator:
    """Iterator for a single audio stream from the batch."""
    
    def __init__(self, streamer: AudioStreamer):
        self.streamer = streamer
        
    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.streamer.audio_queues.get(timeout=self.streamer.timeout)
        if value == self.streamer.stop_signal:
            raise StopIteration()
        return value


class AudioBatchIterator:
    """Iterator that yields audio chunks for all samples in the batch."""
    
    def __init__(self, streamer: AudioStreamer):
        self.streamer = streamer
        self.active_samples = set([0])
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.active_samples:
            raise StopIteration()
            
        batch_chunks = {}
        samples_to_remove = set()
        
        # Try to get chunks from all active samples
        try:
            value = self.streamer.audio_queues.get(block=False)
            if value == self.streamer.stop_signal:
                samples_to_remove.add(0)
            else:
                batch_chunks[0] = value
        except:
            # Queue is empty for this sample, skip it this iteration
            pass
        
        # Remove finished samples
        self.active_samples -= samples_to_remove
        
        if batch_chunks:
            return batch_chunks
        elif self.active_samples:
            # If no chunks were ready but we still have active samples, 
            # wait a bit and try again
            import time
            time.sleep(0.01)
            return self.__next__()
        else:
            raise StopIteration()


class AsyncAudioStreamer(AudioStreamer):
    """
    Async version of AudioStreamer for use in async contexts.
    """
    
    def __init__(
        self, 
        stop_signal: Optional[any] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(stop_signal, timeout)
        # Replace regular queues with async queues
        self.audio_queues = asyncio.Queue()
        self.loop = asyncio.get_running_loop()
        
    def put(self, audio_chunks: torch.Tensor):
        """Put audio chunks in the appropriate async queues."""
        audio_chunk = audio_chunks.detach().cpu()
        self.loop.call_soon_threadsafe(
            self.audio_queues.put_nowait, audio_chunk
        )
    
    def end(self):
        """Signal the end of generation for specified samples."""
            
        if not self.finished_flags:
            self.loop.call_soon_threadsafe(
                self.audio_queues.put_nowait, self.stop_signal
            )
            self.finished_flags = True
    
    async def get_stream(self):
        """Get async iterator for a specific sample's audio stream."""
            
        while True:
            value = await self.audio_queues.get()
            if value == self.stop_signal:
                break
            yield value
    
    def __aiter__(self):
        """Returns an async iterator over all audio streams."""
        return AsyncAudioBatchIterator(self)


class AsyncAudioBatchIterator:
    """Async iterator for batch audio streaming."""
    
    def __init__(self, streamer: AsyncAudioStreamer):
        self.streamer = streamer
        self.active_samples = set([0])
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if not self.active_samples:
            raise StopAsyncIteration()
            
        batch_chunks = {}
        samples_to_remove = set()
        
        # Create tasks for all active samples
        tasks = {
            idx: asyncio.create_task(self._get_chunk(idx)) 
            for idx in self.active_samples
        }
        
        # Wait for at least one chunk to be ready
        done, pending = await asyncio.wait(
            tasks.values(), 
            return_when=asyncio.FIRST_COMPLETED,
            timeout=self.streamer.timeout
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            
        # Process completed tasks
        for idx, task in tasks.items():
            if task in done:
                try:
                    value = await task
                    if value == self.streamer.stop_signal:
                        samples_to_remove.add(idx)
                    else:
                        batch_chunks[idx] = value
                except asyncio.CancelledError:
                    pass
                    
        self.active_samples -= samples_to_remove
        
        if batch_chunks:
            return batch_chunks
        elif self.active_samples:
            # Try again if we still have active samples
            return await self.__anext__()
        else:
            raise StopAsyncIteration()
    
    async def _get_chunk(self, idx):
        """Helper to get a chunk from a specific queue."""
        return await self.streamer.audio_queues.get()