from __future__ import annotations


def register_yolo_custom_layers() -> None:
    """Expose local modules to Ultralytics' YAML parser and checkpoint loader."""
    try:
        import ultralytics.nn.tasks as tasks
    except ImportError as exc:
        raise SystemExit("Please install ultralytics before using custom detection models.") from exc

    from .custom_layers import KANBottleneck, KANC3k2, KANConv, MSPA, SwinTinyLayer

    tasks.KANConv = KANConv
    tasks.KANBottleneck = KANBottleneck
    tasks.KANC3k2 = KANC3k2
    tasks.MSPA = MSPA
    tasks.SwinTinyLayer = SwinTinyLayer
