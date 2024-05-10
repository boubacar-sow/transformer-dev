import os
from io import BufferedReader, BytesIO
from pathlib import Path

import zstandard as zstd

from transformer.utils.config import get_config

cfg, _ = get_config()


class FileSystem:
    def exists(self, file: str) -> bool:
        return os.path.exists(file)

    def ls(self, prefix: str = ".") -> list[str]:
        """
        List objects in the data directory.

        (Code borrowed from lightflow S3Hook)
        """
        keys = []
        for root, dirs, files in os.walk(prefix):
            for name in files + dirs:
                keys.append(os.path.join(root, name))
        return keys

    def rm(self, file: str) -> None:
        """Removes a file."""
        os.remove(file)

    def read_buffer(self, file: str, zstd_format: bool = False) -> BufferedReader | zstd.ZstdDecompressionReader:
        """
        Returns a BytesIO object for the requested file.
        This buffer can be fed, e.g. in pd.read_csv or json.load
        """
        if zstd_format:
            return self._read_buffer_zstd(file=file)
        else:
            return open(file, mode="rb")

    def save_buffer(self, buffer: BytesIO, file: str, zstd_format: bool = False) -> None:
        """
        Reads and save the content of the given bytes buffer.
        """
        self.save_bytes(buffer.getvalue(), file, zstd_format=zstd_format)

    def read_bytes(self, file: str, zstd_format: bool = False) -> bytes:
        if zstd_format:
            return self._read_buffer_zstd(file=file).read()
        else:
            return self.read_buffer(file).read()

    def save_bytes(self, obj: bytes, file: str, zstd_format: bool = False) -> None:
        if zstd_format:
            buffer, writer = self._new_zstd_writer()
            writer.write(obj)
            writer.flush(zstd.FLUSH_FRAME)
            obj = buffer.getvalue()  # get the compressed bytes
            buffer.close()

        path = Path(file)
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)  # Ensure the parent folders exist
        f = open(path, mode="wb")
        f.write(obj)
        f.close()

    def _read_buffer_zstd(self, file: str) -> zstd.ZstdDecompressionReader:
        """
        Returns a read-only, buffer-like object for a zstandard-compressed file.
        """
        buffer = self.read_buffer(file)
        dctx = zstd.ZstdDecompressor()
        zstd_buffer = dctx.stream_reader(buffer, closefd=True)
        return zstd_buffer

    def _new_zstd_writer(self) -> tuple[BytesIO, zstd.ZstdCompressionWriter]:
        """
        Returns a write-only, buffer-like object for a zstandard-compressed file.
        """
        buffer = BytesIO()
        ctx = zstd.ZstdCompressor(level=cfg.data.zst_compression_level)
        zstd_buffer = ctx.stream_writer(buffer, closefd=True)
        return buffer, zstd_buffer
