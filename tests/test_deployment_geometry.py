import unittest

from sapai_gym1.deployment import (
    CANONICAL_CANVAS,
    CanvasSpec,
    NormalizedPoint,
    NormalizedRect,
    ViewportTransform,
)


class TestDeploymentGeometry(unittest.TestCase):
    def test_canonical_canvas_is_full_hd_sixteen_by_nine(self):
        self.assertEqual(
            (CANONICAL_CANVAS.width, CANONICAL_CANVAS.height),
            (1920, 1080),
        )
        self.assertAlmostEqual(CANONICAL_CANVAS.aspect_ratio, 16 / 9)

    def test_normalized_coordinates_scale_exactly_to_1440p(self):
        transform = ViewportTransform.full_viewport(CanvasSpec(2560, 1440))
        self.assertEqual(
            transform.point_to_pixels(NormalizedPoint(0.5, 0.5)),
            (1280, 720),
        )
        self.assertEqual(
            transform.rect_to_pixels(NormalizedRect(0.25, 0.25, 0.5, 0.5)),
            (640, 360, 1280, 720),
        )

    def test_sixteen_by_ten_is_letterboxed_not_stretched(self):
        transform = ViewportTransform.fitted(CanvasSpec(1920, 1200))
        self.assertEqual(
            (
                transform.content_x,
                transform.content_y,
                transform.content_width,
                transform.content_height,
            ),
            (0.0, 60.0, 1920.0, 1080.0),
        )
        self.assertEqual(
            transform.point_to_pixels(NormalizedPoint(0.5, 0.5)),
            (960, 600),
        )

    def test_invalid_normalized_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            NormalizedPoint(1.1, 0.5)
        with self.assertRaises(ValueError):
            NormalizedRect(0.8, 0.2, 0.3, 0.2)


if __name__ == "__main__":
    unittest.main()
