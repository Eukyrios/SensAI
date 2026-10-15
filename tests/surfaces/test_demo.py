# Test for the demo script.

import pytest
from sensai.surfaces.demo import run_demo


@pytest.mark.asyncio
async def test_run_demo():
    await run_demo()
