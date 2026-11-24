"""Container binario gapcomp/1 con i 3 stream compressi separatamente e verifica SHA-256."""
import hashlib
import json
import struct

from compression.codecs import compress_bytes, decompress_bytes

FORMAT_VERSION = "gapcomp/1"
STREAM_NAMES = ("skeleton", "manifest_or_placeholder", "corrections")
FORBIDDEN_KEYS = {"removed_words", "removed_text", "oracle_text", "oracle"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_manifest_no_oracle(manifest_bytes: bytes) -> bool:
    """Verifica che il manifest non contenga mai campi oracle vietati dal protocollo."""
    text = manifest_bytes.decode("utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if any(k in obj for k in FORBIDDEN_KEYS):
            return False
    return True


def pack_container(
    doc_id: str,
    strategy: str,
    ratio: float,
    condition: str,
    model_id: str,
    algorithm: str,
    skeleton_bytes: bytes,
    manifest_bytes: bytes,
    corrections_bytes: bytes,
) -> tuple[bytes, dict]:
    """Comprime i 3 stream e impacchetta il container binario gapcomp/1."""
    if not check_manifest_no_oracle(manifest_bytes):
        raise ValueError("Violazione protocollo: campi oracle vietati trovati nel manifest!")

    raw_streams = {
        "skeleton": skeleton_bytes,
        "manifest_or_placeholder": manifest_bytes,
        "corrections": corrections_bytes,
    }
    comp_streams: dict[str, bytes] = {}
    streams_meta: dict[str, dict] = {}

    for name in STREAM_NAMES:
        raw = raw_streams[name]
        comp = compress_bytes(raw, algorithm)
        comp_streams[name] = comp
        streams_meta[name] = {
            "codec": algorithm,
            "raw_len": len(raw),
            "comp_len": len(comp),
            "raw_sha256": _sha256(raw),
            "comp_sha256": _sha256(comp),
        }

    header = {
        "format": FORMAT_VERSION,
        "doc_id": doc_id,
        "strategy": strategy,
        "ratio_target": ratio,
        "word_selection_condition": condition,
        "model_id": model_id,
        "algorithm": algorithm,
        "streams": streams_meta,
    }
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    blob = (
        struct.pack(">I", len(header_bytes))
        + header_bytes
        + comp_streams["skeleton"]
        + comp_streams["manifest_or_placeholder"]
        + comp_streams["corrections"]
    )
    return blob, header


def unpack_container(blob: bytes) -> tuple[dict, dict[str, bytes], dict[str, bool]]:
    """Decomprime il container binario e verifica l'integrità bit-a-bit di ogni stream."""
    if len(blob) < 4:
        raise ValueError("Container corrotto: troppo corto.")
    header_len = struct.unpack(">I", blob[:4])[0]
    header = json.loads(blob[4 : 4 + header_len].decode("utf-8"))

    if header.get("format") != FORMAT_VERSION:
        raise ValueError(f"Formato container non supportato: {header.get('format')!r}")
    if any(k in header for k in FORBIDDEN_KEYS):
        raise ValueError("Violazione protocollo: campo oracle trovato nell'header!")

    offset = 4 + header_len
    raw_streams: dict[str, bytes] = {}
    checks: dict[str, bool] = {}

    for name in STREAM_NAMES:
        meta = header["streams"][name]
        comp_len = meta["comp_len"]
        comp = blob[offset : offset + comp_len]
        offset += comp_len

        raw = decompress_bytes(comp, meta["codec"])
        ok = (
            len(comp) == comp_len
            and _sha256(comp) == meta["comp_sha256"]
            and len(raw) == meta["raw_len"]
            and _sha256(raw) == meta["raw_sha256"]
        )
        raw_streams[name] = raw
        checks[name] = ok

    if offset != len(blob):
        raise ValueError(f"Byte residui in coda al container: {len(blob) - offset}")

    return header, raw_streams, checks
