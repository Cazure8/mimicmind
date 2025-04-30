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


import copy
import asyncio
import threading
import json
import os
import bittensor as bt
import numpy as np
import argparse

from typing import List, Union, Dict, Any, Tuple
from traceback import print_exception

from mimicmind.utils.misc import update_repository
from mimicmind.base.neuron import BaseNeuron
from mimicmind.base.utils.weight_utils import process_weights_for_netuid, convert_weights_and_uids_for_emit
from mimicmind.utils.config import add_validator_args

class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for MimicMind validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)
        
    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        
        # Initialize performance history for tracking miner performance over time
        self.performance_history: Dict[int, List[Dict[str, Any]]] = {}
        self.history_size = 100  # Maximum records per miner
        
        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def concurrent_forward(self):
        """Run multiple forward passes concurrently."""
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the validator on the Bittensor network.
        """
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()
                
                if self.step % 5 == 0:
                    update_repository()
                    
                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores.
        """
        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )
            # Replace NaN with zeros
            self.scores = np.nan_to_num(self.scores, nan=0.0)

        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def redistribute_rewards_by_rank(self, rewards: np.ndarray, uids: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Redistributes rewards based on performance ranking:
        - 1st place: 50% of rewards
        - 2nd place: 25% of rewards
        - 3rd place: 15% of rewards
        - Remaining miners: Share 10% of rewards proportionally
        
        Args:
            rewards: Original reward values
            uids: Corresponding UIDs for each reward
            
        Returns:
            Tuple containing redistributed rewards array and rank information dictionary
        """
        # Handle edge cases
        if len(rewards) == 0:
            return rewards, {}
            
        if len(rewards) == 1:
            return np.array([1.0]), {"ranks": {int(uids[0]): 1}}
        
        # Pair UIDs with their rewards for sorting
        uid_reward_pairs = [(uid, reward) for uid, reward in zip(uids, rewards)]
        
        # Sort by reward in descending order
        sorted_pairs = sorted(uid_reward_pairs, key=lambda x: x[1], reverse=True)
        sorted_uids = [pair[0] for pair in sorted_pairs]
        sorted_rewards = [pair[1] for pair in sorted_pairs]
        
        # Initialize redistributed rewards
        redistributed = np.zeros_like(rewards)
        rank_info = {"ranks": {}, "original_rewards": {}, "new_rewards": {}}
        
        # Create a mapping from original index to sorted index
        uid_to_index = {uid: i for i, uid in enumerate(uids)}
        
        # Set up the tiered distribution percentages
        total_reward = 1.0  
        tier_percentages = [0.5, 0.25, 0.15]  
        
        # Distribute rewards for top 3 positions 
        for rank, percentage in enumerate(tier_percentages):
            if rank < len(sorted_uids):
                uid = sorted_uids[rank]
                original_idx = uid_to_index[uid]
                redistributed[original_idx] = total_reward * percentage
                
                # Store info for logging
                rank_info["ranks"][int(uid)] = rank + 1
                rank_info["original_rewards"][int(uid)] = float(rewards[original_idx])
                rank_info["new_rewards"][int(uid)] = float(redistributed[original_idx])
        
        # Calculate remaining reward pool 
        remaining_pool = total_reward * 0.1
        
        # Handle remaining miners 
        if len(sorted_uids) > 3:
            remaining_uids = sorted_uids[3:]
            remaining_rewards = sorted_rewards[3:]
            
            # Normalize remaining original rewards for proportional distribution
            total_remaining_original = sum(remaining_rewards)
            
            # Distribute remaining 10% pool proportionally
            if total_remaining_original > 0:
                for uid, reward in zip(remaining_uids, remaining_rewards):
                    original_idx = uid_to_index[uid]
                    proportion = reward / total_remaining_original
                    redistributed[original_idx] = remaining_pool * proportion
                    
                    # Store info for logging
                    rank_info["ranks"][int(uid)] = "shared pool"  # Not a numbered rank
                    rank_info["original_rewards"][int(uid)] = float(rewards[original_idx])
                    rank_info["new_rewards"][int(uid)] = float(redistributed[original_idx])
            else:
                # If all remaining rewards are 0, distribute equally
                equal_share = remaining_pool / len(remaining_uids) if len(remaining_uids) > 0 else 0
                for uid in remaining_uids:
                    original_idx = uid_to_index[uid]
                    redistributed[original_idx] = equal_share
                    
                    # Store info for logging
                    rank_info["ranks"][int(uid)] = "shared pool"
                    rank_info["original_rewards"][int(uid)] = float(rewards[original_idx])
                    rank_info["new_rewards"][int(uid)] = float(redistributed[original_idx])
        
        return redistributed, rank_info

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """
        Performs exponential moving average on the scores based on the rewards received from the miners.
        Redistributes rewards based on performance ranking before updating scores.
        Also tracks performance history for analytics and anti-gaming mechanisms.
        """
        bt.logging.info("Updating scores with new rewards")
        
        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)
        
        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )
        
        # Log original rewards for reference
        bt.logging.info("Original rewards: " + ", ".join([f"UID {uid}: {reward:.4f}" for uid, reward in zip(uids_array, rewards)]))
        
        # Redistribute rewards according to ranking
        redistributed_rewards, rank_info = self.redistribute_rewards_by_rank(rewards, uids_array)
        
        # Log redistributed rewards
        bt.logging.info("===== PERFORMANCE RANKING =====")
        for uid, rank in sorted(rank_info["ranks"].items(), key=lambda x: x[1] if isinstance(x[1], int) else 999):
            if isinstance(rank, int):
                bt.logging.info(f"Rank #{rank}: UID {uid} - Original: {rank_info['original_rewards'][uid]:.4f}, New: {rank_info['new_rewards'][uid]:.4f}")
        
        if "shared pool" in rank_info["ranks"].values():
            bt.logging.info("--- Shared Pool (10%) ---")
            for uid, rank in rank_info["ranks"].items():
                if rank == "shared pool":
                    bt.logging.info(f"UID {uid} - Original: {rank_info['original_rewards'][uid]:.4f}, New: {rank_info['new_rewards'][uid]:.4f}")
        bt.logging.info("=============================")
            
        # Record performance in history for analytics and anti-gaming
        timestamp = asyncio.get_event_loop().time()
        for i, uid in enumerate(uids_array):
            uid_int = int(uid)
            original_reward = float(rewards[i])
            redistributed_reward = float(redistributed_rewards[i])
            
            # Initialize history for this uid if not exists
            if uid_int not in self.performance_history:
                self.performance_history[uid_int] = []
                
            # Add performance data
            performance_data = {
                "timestamp": timestamp,
                "original_reward": original_reward,
                "redistributed_reward": redistributed_reward,
                "rank": rank_info["ranks"].get(uid_int, "unranked"),
                "step": self.step
            }
            
            # Append new data and trim if needed
            self.performance_history[uid_int].append(performance_data)
            if len(self.performance_history[uid_int]) > self.history_size:
                self.performance_history[uid_int] = self.performance_history[uid_int][-self.history_size:]
        
        # Compute forward pass rewards using redistributed rewards, assumes uids are mutually exclusive.
        scattered_rewards = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = redistributed_rewards

        alpha = self.config.neuron.moving_average_alpha

        self.scores = (alpha * scattered_rewards) + (1 - alpha) * self.scores
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
        )
        
        # Save performance history to a JSON file 
        try:
            # Convert keys to strings 
            history_dict = {str(k): v for k, v in self.performance_history.items()}
            
            with open(os.path.join(self.config.neuron.full_path, "performance_history.json"), "w") as f:
                json.dump(history_dict, f)
        except Exception as e:
            bt.logging.warning(f"Failed to save performance history: {e}")

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        try:
            # Load the state of the validator from file.
            state = np.load(self.config.neuron.full_path + "/state.npz", allow_pickle=True)
            self.step = state["step"]
            self.scores = state["scores"]
            self.hotkeys = state["hotkeys"]
            
            # Try to load performance history
            history_path = os.path.join(self.config.neuron.full_path, "performance_history.json")
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history_dict = json.load(f)
                    # Convert keys back to integers
                    self.performance_history = {int(k): v for k, v in history_dict.items()}
        except Exception as e:
            bt.logging.warning(f"Failed to load state: {e}. Starting with a fresh state.")