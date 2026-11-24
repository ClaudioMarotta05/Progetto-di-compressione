"""Test unitari Fase 6: codec, residui, container gapcomp/1, ricostruzione lossless e divieto oracle."""
from pathlib import Path
import json

from compression.codecs import ALGORITHMS, compress_bytes, decompress_bytes
from compression.container import check_manifest_no_oracle, pack_container, unpack_container
from compression.residuals import decode_residuals, encode_residuals, reconstruct_text
from elision.generator import generate, load_config

BASE = Path(".")
SAMPLE = (
    "Cariacica è un comune del Brasile nello Stato dell'Espírito Santo.\n"
    "Il testo contiene accenti, punteggiatura (parentesi) e backslash \\ e ritorni a capo.\n"
) * 5


def test_all_codecs_roundtrip():
    payload = SAMPLE.encode("utf-8")
    for algo in ALGORITHMS:
        comp = compress_bytes(payload, algo)
        assert decompress_bytes(comp, algo) == payload


def test_residuals_and_container_lossless_A_and_B():
    config = load_config(BASE / "config/base.yaml")
    elision_cfg = load_config(BASE / "config/elision.yaml")
    condition = elision_cfg["gap_generation"]["word_selection"]["conditions"][0]

    for strategy in ("A", "B"):
        skeleton, manifest, oracle, _ = generate(
            text=SAMPLE,
            doc_id="test_doc",
            strategy=strategy,
            ratio=0.30,
            condition=condition,
            config=config,
            base_dir=BASE,
        )
        fake_preds = []
        for idx, (m, o) in enumerate(zip(manifest, oracle)):
            words = o["removed_text"].strip().split()
            inferred = words if idx % 2 == 0 else ["parola_sbagliata"]
            fake_preds.append({"gap_id": m["gap_id"], "inferred": inferred})

        residuals_bytes, stats = encode_residuals(manifest, fake_preds, oracle)
        assert stats["n_gaps"] == len(manifest)

        manifest_bytes = "\n".join(json.dumps(m) for m in manifest).encode("utf-8")
        assert check_manifest_no_oracle(manifest_bytes)

        for algo in ALGORITHMS:
            blob, _ = pack_container(
                doc_id="test_doc",
                strategy=strategy,
                ratio=0.30,
                condition=condition.get("name", "test_cond"),
                model_id="test_model",
                algorithm=algo,
                skeleton_bytes=skeleton.encode("utf-8"),
                manifest_bytes=manifest_bytes,
                corrections_bytes=residuals_bytes,
            )
            _, dec_streams, checks = unpack_container(blob)
            assert all(checks.values())

            dec_manifest = [
                json.loads(line)
                for line in dec_streams["manifest_or_placeholder"].decode("utf-8").splitlines()
                if line.strip()
            ]
            gap_texts = decode_residuals(dec_streams["corrections"], dec_manifest, fake_preds)
            rebuilt = reconstruct_text(
                strategy, dec_streams["skeleton"].decode("utf-8"), dec_manifest, gap_texts
            )
            assert rebuilt == SAMPLE


def test_manifest_rejects_oracle_fields():
    bad_manifest = json.dumps({"gap_id": 0, "removed_text": "segreto"}).encode("utf-8")
    assert not check_manifest_no_oracle(bad_manifest)
