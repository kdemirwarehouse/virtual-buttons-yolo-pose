"""Unit tests for pure helper functions in virtual_buttons."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.virtual_buttons import point_in_rect


class TestPointInRect:
    def test_center_inside(self):
        assert point_in_rect((95, 75), (30, 30, 160, 120)) is True

    def test_top_left_corner(self):
        assert point_in_rect((30, 30), (30, 30, 160, 120)) is True

    def test_bottom_right_corner(self):
        assert point_in_rect((160, 120), (30, 30, 160, 120)) is True

    def test_outside_right(self):
        assert point_in_rect((161, 75), (30, 30, 160, 120)) is False

    def test_outside_top(self):
        assert point_in_rect((95, 29), (30, 30, 160, 120)) is False

    def test_outside_left(self):
        assert point_in_rect((29, 75), (30, 30, 160, 120)) is False

    def test_outside_bottom(self):
        assert point_in_rect((95, 121), (30, 30, 160, 120)) is False

    def test_origin_outside(self):
        assert point_in_rect((0, 0), (30, 30, 160, 120)) is False
