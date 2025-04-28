# The MIT License (MIT)
# Copyright © 2025 Cazure

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
import base64
import bittensor as bt
import torch

from mimicmind.protocol import MimicMindSynapse
from mimicmind.validator.clone_reward import get_clone_rewards
from mimicmind.utils.uids import get_random_uids
from mimicmind.utils.helper import fetch_random_sentences, get_random_audio_clip

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    """
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    try:
        # Get voice clip and text for cloning
        clone_text = fetch_random_sentences()
        print(f"clone_text: {clone_text}")
        
        clone_clip_path = get_random_audio_clip()
        print(f"clone_audio_path: {clone_clip_path}")
        
        with open(clone_clip_path, "rb") as audio_file:
            clone_clip = audio_file.read() 

        clone_clip_base64 = base64.b64encode(clone_clip).decode("utf-8")
        
        # Query miners for voice cloning
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=MimicMindSynapse(clone_clip=clone_clip_base64, clone_text=clone_text),
            deserialize=False,
            timeout=50
        )
        
        # Calculate rewards based on responses
        rewards = get_clone_rewards(self, clip_audio_path=clone_clip_path, clone_text=clone_text, responses=responses)
        self.update_scores(rewards, miner_uids)
    
    except Exception as e:
        bt.logging.error(f"An error occurred: {e}")
        rewards = torch.zeros(len(miner_uids))
        
    bt.logging.info(f"Scored responses: {rewards}")