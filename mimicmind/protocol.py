import typing
import bittensor as bt

class MimicMindSynapse(bt.Synapse):
    """
    Protocol representation for handling voice cloning requests and responses in the MimicMind subnet.
    
    This protocol handles voice cloning functionality:
    - Accepts a voice clip and text to generate a cloned audio output.
    """
    # Fields for Voice Cloning
    clone_clip: typing.Optional[str] = None  # Audio clip data as base64 encoded string
    clone_text: typing.Optional[str] = None  # Text to be synthesized
    clone_audio: typing.Optional[str] = None  # Generated cloned audio as base64 encoded string

    def deserialize(self) -> typing.Optional[str|bytes]:
        """
        Deserialize and return the cloned audio.
        """
        return self.clone_audio.decode() if self.clone_audio else None