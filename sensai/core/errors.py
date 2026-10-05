from typing import Literal

# Error categories shared by the transport, the turn loop and the surfaces.
ErrorKind = Literal["model_missing", "connection", "bad_request", "stream", "timeout", "empty_input",
                    "invalid_output"]       # a structured reply that still fails its schema after one repair
