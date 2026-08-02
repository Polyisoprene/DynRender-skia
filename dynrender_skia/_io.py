"""Image fetching — shared HTTP connection pool and request helpers."""

import asyncio
import threading
from typing import Optional, Union

import httpx
import skia
from loguru import logger

_IMG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://t.bilibili.com/",
    "Origin": "https://t.bilibili.com",
}

_client: Optional[httpx.AsyncClient] = None
_client_loop: Optional[asyncio.AbstractEventLoop] = None
_client_lock = threading.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return the shared HTTP client for the current event loop (lazy init, thread-safe).

    httpcore pools connections inside the client, and each pooled connection is
    bound to the event loop it was created on. Reusing a client created on a
    different (possibly closed) loop raises "Event loop is closed", so the
    client is rebuilt whenever the running loop changes.
    """
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is not None and not _client.is_closed and _client_loop is loop:
        return _client
    with _client_lock:  # creation is synchronous (no I/O), safe to hold across threads
        if _client is not None and not _client.is_closed and _client_loop is loop:
            return _client
        # Drop the stale client instead of closing it: a closed loop has already
        # cleaned up its transports, and closing it here could itself raise
        # "Event loop is closed" or race an in-flight render on a live loop.
        _client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=5),
            headers=_IMG_HEADERS,
            timeout=httpx.Timeout(30.0),
        )
        _client_loop = loop
        return _client


async def fetch_images(
    url: Union[str, list[str]], size: Optional[tuple[int, int]] = None, retries: int = 5
) -> Union[skia.Image, tuple[skia.Image, ...]]:
    """Fetch image(s) from URL(s), optionally resizing.

    Uses a shared HTTP client for connection pooling — orders of magnitude
    faster than creating a new client per call.
    """
    client = await _get_client()
    if isinstance(url, list):
        return await asyncio.gather(
            *[_request_img(client, u, size) for u in url]
        )
    return await _request_img(client, url, size)


async def _request_img(
    client: httpx.AsyncClient, url: str, size: Optional[tuple[int, int]],
) -> Optional[skia.Image]:
    try:
        response = await client.get(url)
        img: skia.Image = skia.Image.MakeFromEncoded(response.content)  # type: ignore
        if img is None:
            logger.error("Image decode error or request returned none in content")
        return img.resize(*size) if size is not None else img
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.exception(f"Request or HTTP error occurred: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return None


request_img = _request_img
