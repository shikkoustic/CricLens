"""Stream members out of a ZIP served over HTTP without storing the whole archive.

Dropbox serves CricketVision's 24 GB Videos.zip as an on-the-fly archive: no HTTP
range support, and members are *stored* (uncompressed) with a trailing data
descriptor, so the local header carries no size. Generic stream unzippers refuse
that layout. We find each member's end by scanning for the data-descriptor
signature and accepting it only when the CRC-32 and byte count that follow match
the bytes read so far (a false match is astronomically unlikely).
"""
from __future__ import annotations

import struct
import zlib
from typing import BinaryIO, Callable, Iterable, Iterator

LOCAL = b"PK\x03\x04"
DESC = b"PK\x07\x08"
CENTRAL = b"PK\x01\x02"


class _Reader:
    def __init__(self, chunks: Iterable[bytes]):
        self._it = iter(chunks)
        self.buf = bytearray()

    def fill(self, n: int) -> bool:
        while len(self.buf) < n:
            try:
                self.buf += next(self._it)
            except StopIteration:
                return False
        return True

    def take(self, n: int) -> bytes:
        if not self.fill(n):
            raise EOFError("archive truncated")
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out


def iter_members(chunks: Iterable[bytes], open_out: Callable[[str], BinaryIO | None]) -> Iterator[tuple[str, int]]:
    """Walk the archive; for each member call open_out(name) -> writable file or None (skip).

    Yields (name, size) after each member is fully consumed.
    """
    r = _Reader(chunks)
    while True:
        if not r.fill(4) or bytes(r.buf[:4]) == CENTRAL:
            return
        if bytes(r.buf[:4]) != LOCAL:
            raise ValueError(f"unexpected signature {bytes(r.buf[:4])!r}")
        hdr = r.take(30)
        _, _, flags, method, _, _, crc, csize, usize, nlen, xlen = struct.unpack("<IHHHHHIIIHH", hdr)
        name = r.take(nlen).decode("utf-8", "replace")
        extra = r.take(xlen)
        zip64 = _has_zip64(extra)
        out = open_out(name) if not name.endswith("/") else None
        if flags & 0x08:
            size = _stored_until_descriptor(r, out, zip64) if method == 0 else _deflated(r, out, zip64, has_desc=True)
        else:
            if method == 0:
                size = _copy(r, out, csize)
            else:
                size = _deflated(r, out, zip64, has_desc=False)
        if out is not None:
            out.close()
        yield name, size


def _has_zip64(extra: bytes) -> bool:
    i = 0
    while i + 4 <= len(extra):
        tag, ln = struct.unpack("<HH", extra[i:i + 4])
        if tag == 0x0001:
            return True
        i += 4 + ln
    return False


def _copy(r: _Reader, out, n: int) -> int:
    left = n
    while left:
        if not r.buf and not r.fill(1):
            raise EOFError("archive truncated")
        k = min(left, len(r.buf))
        if out is not None:
            out.write(r.buf[:k])
        del r.buf[:k]
        left -= k
    return n


def _stored_until_descriptor(r: _Reader, out, zip64: bool) -> int:
    crc, written = 0, 0
    tail = 24  # keep enough bytes to recognise a descriptor split across chunks
    while True:
        start = 0
        while True:
            i = r.buf.find(DESC, start)
            if i < 0:
                break
            need = i + 4 + (20 if zip64 else 12)
            if len(r.buf) < need and not r.fill(need):
                break
            c = zlib.crc32(r.buf[:i], crc)
            d_crc = struct.unpack_from("<I", r.buf, i + 4)[0]
            if zip64:
                d_size = struct.unpack_from("<Q", r.buf, i + 8)[0]
            else:
                d_size = struct.unpack_from("<I", r.buf, i + 8)[0]
            if d_crc == c and d_size == written + i:
                if out is not None:
                    out.write(r.buf[:i])
                del r.buf[:need]
                return written + i
            start = i + 1
        if len(r.buf) > tail:
            k = len(r.buf) - tail
            crc = zlib.crc32(r.buf[:k], crc)
            if out is not None:
                out.write(r.buf[:k])
            written += k
            del r.buf[:k]
        if not r.fill(len(r.buf) + (1 << 20)) and DESC not in r.buf:
            raise EOFError("no data descriptor before end of stream")


def _deflated(r: _Reader, out, zip64: bool, has_desc: bool) -> int:
    d = zlib.decompressobj(-15)
    n = 0
    while not d.eof:
        if not r.buf and not r.fill(1):
            raise EOFError("archive truncated")
        data = d.decompress(bytes(r.buf))
        r.buf = bytearray(d.unused_data)
        if out is not None:
            out.write(data)
        n += len(data)
    if has_desc:
        r.fill(4)
        if bytes(r.buf[:4]) == DESC:
            r.take(4)
        r.take(20 if zip64 else 12)
    return n
