from typing import Literal

# Error categories shared by the transport, the turn loop and the surfaces.
ErrorKind = Literal["model_missing", "connection", "bad_request", "stream", "timeout", "empty_input"]
