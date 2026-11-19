from __future__ import annotations

import numpy as np
import pytest

from glakit.color import RGB, RGBA


class TestRGB:
    def test_creation_and_properties(self):
        c1 = RGB((255, 0, 128))
        assert isinstance(c1, RGB)
        assert c1.shape == (3,)
        assert c1.dtype == np.uint8
        assert (c1.r, c1.g, c1.b) == (255, 0, 128)
        assert c1.tuple == (255, 0, 128)

        # From HEX (3 and 6 chars)
        assert RGB("#f08") == RGB((255, 0, 136))
        assert RGB("#ff0080") == RGB((255, 0, 128))

        # From CSS3 name
        red = RGB("red")
        assert red == RGB((255, 0, 0))
        assert str(red) == "red"
        assert red.to_hex() == "#ff0000"

        # From normalized floats [0.0, 1.0]
        c_float = RGB((1.0, 0.5, 0.0))
        assert c_float == RGB((255, 128, 0))

    def test_immutability_and_copy(self):
        c = RGB((10, 20, 30))
        assert not c.flags.writeable
        with pytest.raises(TypeError):
            c[0] = 99  # type: ignore

        arr = c.copy()
        assert type(arr) is np.ndarray
        assert arr.flags.writeable
        arr[0] = 99
        assert c.r == 10

    def test_saturating_arithmetic(self):
        c = RGB((200, 100, 10))
        # Overflow saturates at 255
        c_add = c + 100
        assert isinstance(c_add, RGB)
        assert c_add == RGB((255, 200, 110))

        # Underflow saturates at 0
        c_sub = c - 50
        assert isinstance(c_sub, RGB)
        assert c_sub == RGB((150, 50, 0))

        # Multiplication & division
        c_mul = c * 2
        assert isinstance(c_mul, RGB)
        assert c_mul == RGB((255, 200, 20))

        # In-place operations forbidden
        with pytest.raises(TypeError):
            c += 10  # type: ignore

    def test_alpha_drop_warning_and_conversions(self):
        rgba = RGBA((10, 20, 30, 128))
        with pytest.warns(UserWarning, match="Alpha channel dropped"):
            rgb = RGB(rgba)
        assert isinstance(rgb, RGB)
        assert rgb == RGB((10, 20, 30))

        # Conversion back to RGBA
        rgba_opaque = rgb.as_rgba()
        assert isinstance(rgba_opaque, RGBA)
        assert rgba_opaque == RGBA((10, 20, 30, 255))

        rgba_half = rgb.as_rgba(alpha=0.5)
        assert rgba_half == RGBA((10, 20, 30, 128))

    def test_blend(self):
        black = RGB((0, 0, 0))
        white = RGB((255, 255, 255))
        mid = black.blend(white, alpha=0.5)
        assert isinstance(mid, RGB)
        assert mid == RGB((128, 128, 128))


class TestRGBA:
    def test_creation_and_properties(self):
        c = RGBA((10, 20, 30, 40))
        assert isinstance(c, RGBA)
        assert c.shape == (4,)
        assert (c.r, c.g, c.b, c.a) == (10, 20, 30, 40)
        assert c.tuple == (10, 20, 30, 40)
        assert c.to_hex() == "#0a141e28"

        # 3-channel input defaults to opaque alpha=255
        c_from_rgb = RGBA((10, 20, 30))
        assert c_from_rgb == RGBA((10, 20, 30, 255))

        # 4-char and 8-char HEX
        assert RGBA("#1234") == RGBA((0x11, 0x22, 0x33, 0x44))
        assert RGBA("#11223380") == RGBA((0x11, 0x22, 0x33, 0x80))

    def test_equality_and_hashing(self):
        c1 = RGBA((255, 0, 0, 255))
        c2 = RGBA("red")
        assert c1 == c2
        assert len({c1, c2}) == 1


class TestColorPydanticIntegration:
    def test_pydantic_validation_serialization_and_schema(self):
        pydantic = pytest.importorskip("pydantic")

        class StyleConfig(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(validate_assignment=True)
            stroke: RGB
            fill: RGBA
            fill_no_alpha: RGBA

        cfg = StyleConfig(stroke="#ff0000", fill=(0, 255, 0, 125), fill_no_alpha=(0, 255, 0))
        assert isinstance(cfg.stroke, RGB) and cfg.stroke == RGB((255, 0, 0))
        assert isinstance(cfg.fill, RGBA) and cfg.fill == RGBA((0, 255, 0, 125))
        assert isinstance(cfg.fill_no_alpha, RGBA) and cfg.fill_no_alpha == RGBA((0, 255, 0, 255))

        # Python dump yields fixed-length tuples
        dumped = cfg.model_dump(mode="python")
        assert dumped["stroke"] == "#ff0000"
        assert dumped["fill"] == "#00ff007d"
        assert dumped["fill_no_alpha"] == "#00ff00"

        # JSON round-trip
        reloaded = StyleConfig.model_validate_json(cfg.model_dump_json())
        assert reloaded.stroke == cfg.stroke
        assert reloaded.fill == cfg.fill
        assert reloaded.fill_no_alpha == cfg.fill_no_alpha

        # Validation error on invalid color
        with pytest.raises(pydantic.ValidationError):
            StyleConfig(stroke="not_a_color", fill=(0, 0, 0, 255))  # type: ignore
