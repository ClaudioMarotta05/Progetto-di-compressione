"""Codec lossless congelati dal protocollo: deflate (zlib), bwt (bz2), ppm (pyppmd)."""
import bz2
import zlib
import pyppmd

ALGORITHMS = ("deflate", "bwt", "ppm")


def compress_bytes(data: bytes, algorithm: str) -> bytes:
    if algorithm == "deflate":
        return zlib.compress(data, level=9)
    if algorithm == "bwt":
        return bz2.compress(data, compresslevel=9)
    if algorithm == "ppm":
        return pyppmd.compress(data)
    raise ValueError(f"Algoritmo sconosciuto: {algorithm!r}. Attesi: {ALGORITHMS}")


def decompress_bytes(data: bytes, algorithm: str) -> bytes:
    if algorithm == "deflate":
        return zlib.decompress(data)
    if algorithm == "bwt":
        return bz2.decompress(data)
    if algorithm == "ppm":
        return pyppmd.decompress(data)
    raise ValueError(f"Algoritmo sconosciuto: {algorithm!r}. Attesi: {ALGORITHMS}")
