import os
import tempfile

import requests
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "videos")

_client: Client | None = None


def get_client() -> Client:
    """Get or create the Supabase client (singleton)."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client


def download_video(video_path: str, local_path: str | None = None) -> str:
    """
    Download a video from Supabase Storage to a local file.

    Args:
        video_path: Path in Supabase Storage (e.g., "video/123_file.mp4")
        local_path: Local file path to save to. If None, creates a temp file.

    Returns:
        Local file path where the video was saved.
    """
    client = get_client()

    # Get signed URL for download
    signed_url = client.storage.from_(SUPABASE_STORAGE_BUCKET).create_signed_url(
        video_path, expires_in=3600  # 1 hour
    )

    if "error" in signed_url and signed_url["error"]:
        raise Exception(f"Failed to get signed URL: {signed_url['error']}")

    url = signed_url["signedURL"]

    # Create local path if not provided
    if local_path is None:
        _, ext = os.path.splitext(video_path)
        fd, local_path = tempfile.mkstemp(suffix=ext or ".mp4")
        os.close(fd)

    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return local_path


def upload_video(local_path: str, storage_path: str, content_type: str = "video/mp4") -> dict:
    """
    Upload a video file to Supabase Storage.

    Args:
        local_path: Local file path to upload.
        storage_path: Destination path in Supabase Storage.
        content_type: MIME type of the file.

    Returns:
        Upload response data from Supabase.
    """
    client = get_client()

    with open(local_path, "rb") as f:
        response = client.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
            path=storage_path,
            file=f,
            file_options={"content-type": content_type}
        )

    return response


def get_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Get a signed URL for a file in Supabase Storage.

    Args:
        storage_path: Path in Supabase Storage.
        expires_in: URL expiration time in seconds (default 1 hour).

    Returns:
        Signed URL string.
    """
    client = get_client()

    result = client.storage.from_(SUPABASE_STORAGE_BUCKET).create_signed_url(
        storage_path, expires_in=expires_in
    )

    if "error" in result and result["error"]:
        raise Exception(f"Failed to get signed URL: {result['error']}")

    return result["signedURL"]


def get_public_url(storage_path: str) -> str:
    """
    Get a public URL for a file in Supabase Storage.
    Only works if the bucket is public.

    Args:
        storage_path: Path in Supabase Storage.

    Returns:
        Public URL string.
    """
    client = get_client()
    result = client.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(storage_path)
    return result
