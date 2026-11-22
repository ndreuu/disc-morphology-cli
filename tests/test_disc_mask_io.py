"""Synthetic NIfTI and source-frame geometry checks; no models or patient data."""

import gzip
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is optional")
class DiscMaskIOTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from disc_morphology.mask_io import load_nifti, project_to_frame

        self.np = np
        self.load = load_nifti
        self.project = project_to_frame
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def nifti_bytes(self, array=None, affine=None, endian="<", datatype=4):
        np = self.np
        if array is None:
            array = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        if affine is None:
            affine = np.array([[0, -2, 0, 11], [1.5, 0, 0, -7], [0, 0, 4, 9], [0, 0, 0, 1]])
        types = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}
        dtype = np.dtype(endian + types[datatype])
        header = bytearray(352)
        struct.pack_into(endian + "i", header, 0, 348)
        struct.pack_into(endian + "8h", header, 40, 3, *array.shape, 1, 1, 1, 1)
        struct.pack_into(endian + "2h", header, 70, datatype, dtype.itemsize * 8)
        struct.pack_into(endian + "3f", header, 108, 352, 1, 0)
        header[123] = 2
        struct.pack_into(endian + "h", header, 254, 1)
        struct.pack_into(endian + "12f", header, 280, *np.asarray(affine)[:3].ravel())
        header[344:348] = b"n+1\x00"
        return bytes(header) + array.astype(dtype).tobytes(order="F")

    def save(self, data, compressed=True):
        path = Path(self.temporary.name) / ("synthetic.nii.gz" if compressed else "synthetic.nii")
        path.write_bytes(gzip.compress(data) if compressed else data)
        return path

    def test_affine_and_fortran_voxel_order_roundtrip_both_endians(self):
        np = self.np
        original = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        expected = np.array([[0, -2, 0, 11], [1.5, 0, 0, -7], [0, 0, 4, 9], [0, 0, 0, 1]])
        for endian in ("<", ">"):
            for compressed in (False, True):
                with self.subTest(endian=endian, compressed=compressed):
                    a, affine = self.load(self.save(self.nifti_bytes(original, expected, endian), compressed))
                    np.testing.assert_array_equal(a, original)
                    np.testing.assert_array_equal(affine, expected)
                    self.assertEqual(a[1, 2, 3], 23)

    def test_supported_scalar_datatypes_preserve_values(self):
        for datatype in (2, 4, 8, 16, 64, 256, 512, 768):
            with self.subTest(datatype=datatype):
                a, _ = self.load(self.save(self.nifti_bytes(datatype=datatype)))
                self.np.testing.assert_array_equal(a, self.np.arange(24).reshape((2, 3, 4)))

    def test_axial_projection_lps_ras_spacing_and_zero_padding(self):
        np = self.np
        array = np.arange(6 * 6 * 5).reshape((6, 6, 5)) + 1
        frame = {"rows": 4, "columns": 5, "position": [1, 0, 3],
                 "orientation": [-1, 0, 0, 0, -1, 0], "spacing": [2, 1]}
        actual = self.project(array, np.eye(4), frame)
        expected = np.zeros((4, 5), dtype=array.dtype)
        expected[:3, 1:] = array[0:4, [0, 2, 4], 3].T
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[1, 3], array[2, 2, 3])

    def test_oblique_anisotropic_sagittal_at_known_world_coordinates(self):
        np = self.np
        angle = math.pi / 6
        c = np.array([0, math.cos(angle), math.sin(angle)])
        r = np.array([0, math.sin(angle), -math.cos(angle)])
        n = np.cross(c, r)
        flip = np.array([-1, -1, 1])
        origin_lps = np.array([12, -5, 20])
        affine = np.eye(4)
        affine[:3, :3] = (np.column_stack((c * .5, r * 2, n * 3)).T * flip).T
        affine[:3, 3] = origin_lps * flip
        array = np.arange(5 * 4 * 3).reshape((5, 4, 3))
        frame = {"rows": 4, "columns": 5, "position": (origin_lps + n * 6).tolist(),
                 "orientation": list(c) + list(r), "spacing": [2, .5]}
        np.testing.assert_allclose(affine @ [3, 1, 2, 1],
                                   [*(((origin_lps + c * 1.5 + r * 2 + n * 6) * flip)), 1])
        result = self.project(array, affine, frame)
        np.testing.assert_array_equal(result, array[:, :, 2].T)
        self.assertEqual(result[1, 3], array[3, 1, 2])

    def test_nearest_neighbour_ties_and_outside_footprint(self):
        np = self.np
        frame = {"rows": 1, "columns": 4, "position": [.5, 0, 0],
                 "orientation": [-1, 0, 0, 0, -1, 0], "spacing": [1, 1]}
        array = np.array([7, 9]).reshape((2, 1, 1))
        np.testing.assert_array_equal(self.project(array, np.eye(4), frame), [[7, 9, 0, 0]])
        frame["position"][0] = .501
        self.assertEqual(self.project(array, np.eye(4), frame)[0, 0], 0)

    def test_rejects_malformed_headers_scaling_truncation_and_nonfinite(self):
        original = self.nifti_bytes()
        mutations = []
        for offset, fmt, values in ((0, "i", [0]), (40, "h", [4]), (42, "h", [32767]),
                                    (70, "h", [999]), (72, "h", [32]),
                                    (108, "f", [352.5]), (108, "f", [float("inf")]),
                                    (112, "f", [2]), (112, "f", [float("nan")]),
                                    (116, "f", [1]), (254, "h", [0]),
                                    (280, "f", [float("nan")])):
            data = bytearray(original)
            struct.pack_into("<" + fmt, data, offset, *values)
            mutations.append(bytes(data))
        data = bytearray(original); data[344:348] = b"ni1\x00"; mutations.append(bytes(data))
        data = bytearray(original); data[123] = 1; mutations.append(bytes(data))
        mutations += [original[:100], original[:-1]]
        for data in mutations:
            with self.subTest(header=data[:128]), self.assertRaises(ValueError):
                self.load(self.save(data))
        array = self.np.zeros((2, 3, 4)); array[0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            self.load(self.save(self.nifti_bytes(array, datatype=16)))
        with self.assertRaises(ValueError):
            self.load(self.save(self.nifti_bytes(affine=self.np.zeros((4, 4)))))

    def test_rejects_invalid_projection_geometry_and_source(self):
        np = self.np
        valid = {"rows": 2, "columns": 2, "position": [0, 0, 0],
                 "orientation": [1, 0, 0, 0, 1, 0], "spacing": [1, 1]}
        for patch in ({"spacing": [0, 1]}, {"orientation": [1, 0, 0, 1, 0, 0]},
                      {"rows": 0}, {"columns": True}, {"position": [float("nan"), 0, 0]}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.project(np.ones((2, 2, 2)), np.eye(4), dict(valid, **patch))
        for array in (np.zeros((2, 2)), np.full((2, 2, 2), float("nan"))):
            with self.assertRaises(ValueError):
                self.project(array, np.eye(4), valid)
        with self.assertRaises(ValueError):
            self.project(np.ones((2, 2, 2)), np.zeros((4, 4)), valid)
        with self.assertRaises(ValueError):
            self.project(np.ones((2, 2, 2)), np.eye(4), {})

    def test_rejects_invalid_gzip_stream(self):
        path = Path(self.temporary.name) / "broken.nii.gz"
        for data in (b"not gzip", gzip.compress(self.nifti_bytes())[:20]):
            path.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load(path)


if __name__ == "__main__":
    unittest.main()
