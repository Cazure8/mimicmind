# The MIT License (MIT)
# Copyright © 2025 Your Name

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import base64
import torch
import bittensor as bt
from typing import Dict, List
import asyncio

from mimicmind.protocol import MimicMindSynapse
from mimicmind.validator.clone_reward import get_clone_rewards
from mimicmind.utils.uids import get_random_uids
from mimicmind.utils.helper import fetch_random_sentences, get_random_audio_clip

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses
    based on both quality and latency.
    """
    # Get miners to query
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    miner_uid_list = miner_uids.tolist() if hasattr(miner_uids, 'tolist') else list(miner_uids)

    try:
        # Get voice clip and text for cloning
        clone_text = fetch_random_sentences()
        bt.logging.info(f"Clone text: {clone_text}")
        
        clone_clip_path = get_random_audio_clip()
        bt.logging.info(f"Clone audio path: {clone_clip_path}")
        
        with open(clone_clip_path, "rb") as audio_file:
            clone_clip = audio_file.read() 

        clone_clip_base64 = base64.b64encode(clone_clip).decode("utf-8")
        
        # Create synapse for request
        synapse = MimicMindSynapse(clone_clip=clone_clip_base64, clone_text=clone_text)
        
        # Record request timestamps - record before sending requests
        start_time = time.time()
        synapse.request_timestamp = start_time
        
        # Create dictionary to store request start times for each miner
        request_times: Dict[int, float] = {}
        for uid in miner_uid_list:
            request_times[uid] = start_time
        
        # Send requests to miners
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=synapse,
            deserialize=False,
            timeout=self.config.neuron.timeout
        )
        
        # Process response times and sizes
        end_time = time.time()
        processing_times: Dict[int, float] = {}
        response_sizes: Dict[int, int] = {}
        
        for i, response in enumerate(responses):
            uid = miner_uid_list[i]
            
            # Calculate processing time
            process_time = 0
            if hasattr(response.dendrite, 'process_time') and response.dendrite.process_time is not None:
                # If bittensor provides a process_time, use it
                process_time = response.dendrite.process_time
            else:
                # Otherwise calculate from our recorded timestamps
                process_time = end_time - request_times[uid]
            
            processing_times[uid] = process_time
            
            # Calculate response size
            if response.clone_audio:
                try:
                    audio_bytes = base64.b64decode(response.clone_audio)
                    response_sizes[uid] = len(audio_bytes)
                except Exception as e:
                    bt.logging.error(f"Error decoding response from miner {uid}: {e}")
                    response_sizes[uid] = 0
            else:
                response_sizes[uid] = 0
                
            # Log performance data for debugging
            bt.logging.debug(f"Miner {uid} - Process time: {process_time:.3f}s, Response size: {response_sizes[uid]} bytes")
        
        # Calculate rewards based on both quality and latency
        rewards = get_clone_rewards(
            self, 
            clip_audio_path=clone_clip_path, 
            clone_text=clone_text, 
            responses=responses,
            miner_uids=miner_uid_list,
            processing_times=processing_times,
            response_sizes=response_sizes
        )
        
        # Update validator's scores for miners
        self.update_scores(rewards, miner_uids)
        
    except Exception as e:
        bt.logging.error(f"An error occurred in forward: {e}")
        rewards = torch.zeros(len(miner_uids))
        
    bt.logging.info(f"Scored responses: {rewards}")