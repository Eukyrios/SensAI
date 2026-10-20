# Error categories shared by the transport, the turn loop and the surfaces.

from typing import Literal

ErrorKind = Literal["model_missing", "connection", "bad_request", "stream", "timeout", "empty_input",
                    "invalid_output", "max_hops", "handoff_cycle", "turn_budget"]
