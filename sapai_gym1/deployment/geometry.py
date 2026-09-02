"""Resolution-independent geometry for the live Unity client.

The first supported deployment profile intentionally fixes the browser's
Unity viewport at 1920x1080.  Normalized coordinates still make captures and
input events scale exactly to other 16:9 resolutions.  Non-16:9 windows must
be letterboxed to this logical canvas instead of silently stretching click
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanvasSpec:
    width: int
    height: int
    device_scale_factor: float = 1.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        if self.device_scale_factor <= 0:
            raise ValueError("device_scale_factor must be positive")

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


CANONICAL_CANVAS = CanvasSpec(1920, 1080, 1.0)


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float

    def __post_init__(self):
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ValueError("Normalized point coordinates must be in [0, 1]")


@dataclass(frozen=True)
class NormalizedRect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        values = (self.x, self.y, self.width, self.height)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Normalized rectangle values must be in [0, 1]")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Normalized rectangle must have positive area")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("Normalized rectangle exceeds the canvas")

    @property
    def center(self) -> NormalizedPoint:
        return NormalizedPoint(
            self.x + self.width / 2,
            self.y + self.height / 2,
        )


@dataclass(frozen=True)
class ViewportTransform:
    """Map canonical normalized geometry into a physical screenshot/window."""

    viewport: CanvasSpec
    content_x: float
    content_y: float
    content_width: float
    content_height: float

    @classmethod
    def fitted(
        cls,
        viewport: CanvasSpec,
        canonical: CanvasSpec = CANONICAL_CANVAS,
    ) -> "ViewportTransform":
        """Letterbox a canonical canvas into an arbitrary physical viewport."""

        scale = min(
            viewport.width / canonical.width,
            viewport.height / canonical.height,
        )
        width = canonical.width * scale
        height = canonical.height * scale
        return cls(
            viewport=viewport,
            content_x=(viewport.width - width) / 2,
            content_y=(viewport.height - height) / 2,
            content_width=width,
            content_height=height,
        )

    @classmethod
    def full_viewport(cls, viewport: CanvasSpec) -> "ViewportTransform":
        return cls(viewport, 0.0, 0.0, viewport.width, viewport.height)

    def point_to_pixels(self, point: NormalizedPoint) -> tuple[int, int]:
        return (
            round(self.content_x + point.x * self.content_width),
            round(self.content_y + point.y * self.content_height),
        )

    def rect_to_pixels(
        self, rectangle: NormalizedRect
    ) -> tuple[int, int, int, int]:
        x, y = self.point_to_pixels(
            NormalizedPoint(rectangle.x, rectangle.y)
        )
        x2, y2 = self.point_to_pixels(
            NormalizedPoint(
                rectangle.x + rectangle.width,
                rectangle.y + rectangle.height,
            )
        )
        return x, y, x2 - x, y2 - y

    def pixels_to_point(self, x: float, y: float) -> NormalizedPoint:
        nx = (x - self.content_x) / self.content_width
        ny = (y - self.content_y) / self.content_height
        return NormalizedPoint(nx, ny)

    def is_native_aspect(self, tolerance: float = 1e-3) -> bool:
        return (
            abs(self.content_width / self.content_height - 16 / 9)
            <= tolerance
        )
