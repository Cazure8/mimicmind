import bittensor as bt
import numpy as np
import torch


def process_weights_for_netuid(
    uids: torch.Tensor,
    weights: torch.Tensor,
    netuid: int,
    subtensor: "bt.subtensor",
    metagraph: "bt.metagraph.Metagraph",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process weights for a netuid and return the processed weights.

    Args:
        uids (torch.Tensor): Tensor of uids.
        weights (torch.Tensor): Tensor of weights.
        netuid (int): The netuid to get the weights for.
        subtensor (bt.subtensor): The subtensor to use for the weights.
        metagraph (bt.metagraph.Metagraph): The metagraph to use for the weights.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: uids, weights
    """

    # Filter out delegates and ensure the key exists.
    if subtensor.network == "finney":
        delegates = subtensor.get_delegates(netuid)
        # Convert delegates to UIDs
        delegate_uids = []
        for delegate in delegates:
            if delegate in metagraph.hotkeys:
                delegate_uids.append(metagraph.hotkeys.index(delegate))
            else:
                bt.logging.trace(f"Delegate {delegate} not in metagraph, skipping.")

        # Filter out delegates from weights.
        mask = ~np.isin(uids, delegate_uids)
        weights = weights[mask]
        uids = uids[mask]

    # Set weights for uids that are not serving on network to 0
    mask = torch.zeros_like(weights).bool()
    for idx, uid in enumerate(uids):
        # Validators only set weights for serving miners.
        if not metagraph.axons[uid].is_serving:
            mask[idx] = True

    weights[mask] = 0

    # Set nan weights to 0
    weights[torch.isnan(weights)] = 0

    # Set inf weights to 0
    weights[torch.isinf(weights)] = 0

    # Remove negative weights.
    weights = torch.max(weights, torch.zeros_like(weights))

    # Normalize weights.
    if weights.shape[0] > 0:
        if torch.sum(weights) > 0:
            weights = weights / torch.sum(weights)

    # Check if we have any nans or inf
    if torch.any(torch.isnan(weights)):
        weights = torch.zeros_like(weights)

    if torch.any(torch.isinf(weights)):
        weights = torch.zeros_like(weights)

    # Get the uids of the weights with non-zero values.
    weight_uids = []
    weight_values = []
    for uid, weight in zip(uids, weights):
        if weight > 0:
            weight_uids.append(uid)
            weight_values.append(weight.item())

    # Return the processed weights.
    if len(weight_uids) == 0:
        return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.float32)
    return torch.tensor(weight_uids, dtype=torch.int64), torch.tensor(weight_values, dtype=torch.float32)


def convert_weights_and_uids_for_emit(uids: torch.Tensor, weights: torch.Tensor) -> tuple[list, list]:
    """
    Process weights for emission to the chain.

    Args:
        uids (torch.Tensor): Tensor of uids.
        weights (torch.Tensor): Tensor of weights.

    Returns:
        Tuple[List[int], List[int]]: uids, weights
    """
    # Process into uint16 weights for emission
    if len(uids) == 0:
        return [], []

    # Normalize weights for emission.
    weights = weights / torch.sum(weights)

    # Convert weights to u16 format.
    weights_u16 = (weights * 65535).to(torch.int64)

    # Check if weights are valid.
    if torch.any(weights_u16 < 0):
        bt.logging.warning("Negative weights detected after u16 conversion")
        weights_u16 = torch.max(weights_u16, torch.zeros_like(weights_u16))
    if torch.any(weights_u16 > 65535):
        bt.logging.warning("Weights larger than 65535 detected after u16 conversion")
        weights_u16 = torch.min(weights_u16, 65535 * torch.ones_like(weights_u16))

    # Get non-zero weight uids and values.
    weight_uids = []
    weight_values = []
    for uid, weight in zip(uids, weights_u16):
        if weight > 0:
            weight_uids.append(uid.item())
            weight_values.append(weight.item())

    return weight_uids, weight_values