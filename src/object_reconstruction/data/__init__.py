from .models import CameraFrame, CameraIntrinsics, FramePacket

__all__ = ["CameraFrame", "CameraIntrinsics", "FramePacket", "OfflineFrameSource"]


def __getattr__(name):
    # Lazy export to avoid a circular import: offline_source depends on the
    # calibration package, which in turn imports data.models.
    if name == "OfflineFrameSource":
        from .offline_source import OfflineFrameSource

        return OfflineFrameSource
    raise AttributeError(name)
