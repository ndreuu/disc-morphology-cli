"""Small NIfTI-1 reader and geometric resampling, without inference or registration.

Only single-file 3-D scalar images with an explicit sform are supported. Spatial
units must be millimetres or unspecified; unspecified units require the caller
to confirm millimetres against the source DICOM/conversion provenance. Projection
does not inspect FrameOfReferenceUID and must be gated by the caller. Resampling
a sagittal prediction onto an axial image is not an independent axial prediction.
"""

import gzip
import math
from pathlib import Path
import struct
import zlib

from ._spatial import _axes


_DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8",
           256: "i1", 512: "u2", 768: "u4"}
_MAX_VOXELS = 128 * 1024 * 1024
_MAX_BYTES = 512 * 1024 * 1024
_MAX_DIMENSION = 16384
_MAX_FRAME_PIXELS = 16 * 1024 * 1024


def write_nifti(path, array, affine_ras):
    """Write an unscaled float32 or uint8 volume with an explicit mm sform."""
    import numpy as np
    source = np.asarray(array)
    affine = _affine(affine_ras)
    if source.ndim != 3 or not np.isfinite(source).all() or source.size > _MAX_VOXELS:
        raise ValueError('Expected a bounded finite 3D volume')
    dtype = np.dtype('u1') if source.dtype == np.uint8 else np.dtype('<f4')
    data = np.asarray(source,dtype=dtype)
    header = bytearray(352)
    struct.pack_into('<i',header,0,348)
    struct.pack_into('<8h',header,40,3,*data.shape,1,1,1,1)
    struct.pack_into('<2h',header,70,2 if dtype.itemsize==1 else 16,dtype.itemsize*8)
    steps = np.linalg.norm(affine[:3,:3],axis=0)
    struct.pack_into('<8f',header,76,1,*steps,1,1,1,1)
    struct.pack_into('<f',header,108,352)
    struct.pack_into('<f',header,112,1)
    header[123]=2
    struct.pack_into('<h',header,254,1)
    struct.pack_into('<12f',header,280,*affine[:3].ravel())
    header[344:348]=b'n+1\x00'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if path.name.endswith('.gz') else open
    with opener(path,'wb') as f:
        f.write(header);f.write(data.tobytes(order='F'))


def _affine(value):
    import numpy as np

    try:
        affine = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("A finite nonsingular 4x4 affine is required") from None
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.array_equal(affine[3], [0, 0, 0, 1])):
        raise ValueError("A finite homogeneous 4x4 affine is required")
    linear = affine[:3, :3]
    try:
        condition = float(np.linalg.cond(linear))
    except np.linalg.LinAlgError:
        raise ValueError("Affine is singular") from None
    if not math.isfinite(condition) or condition > 1e12:
        raise ValueError("Affine is singular or ill-conditioned")
    return affine


def load_nifti(path):
    """Return ``(array_xyz, affine_ras)`` from an unscaled .nii or .nii.gz.

    Voxel order is NIfTI order (first index varies fastest in the file), not
    an assumed anatomical XYZ order. Both NIfTI-1 byte orders are supported.
    Scaling other than identity/disabled is rejected, rather than silently
    changing candidate labels. Arrays and affines are finite. No files are
    written and no optional imaging library is required beyond NumPy.
    """
    import numpy as np

    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rb") as stream:
            header = stream.read(352)
            if len(header) != 352 or header[344:348] != b"n+1\x00":
                raise ValueError("Expected a single-file NIfTI-1 header")
            if struct.unpack("<i", header[:4])[0] == 348:
                endian = "<"
            elif struct.unpack(">i", header[:4])[0] == 348:
                endian = ">"
            else:
                raise ValueError("Invalid NIfTI-1 size or byte order")
            dims = struct.unpack(endian + "8h", header[40:56])
            shape = dims[1:4]
            count = math.prod(shape)
            if (dims[0] != 3 or any(n < 1 or n > _MAX_DIMENSION for n in shape)
                    or count > _MAX_VOXELS or any(n not in (0, 1) for n in dims[4:])):
                raise ValueError("Expected a bounded three-dimensional NIfTI image")
            datatype, bitpix = struct.unpack(endian + "2h", header[70:74])
            if datatype not in _DTYPES:
                raise ValueError("Unsupported scalar NIfTI datatype")
            dtype = np.dtype(endian + _DTYPES[datatype])
            if bitpix != dtype.itemsize * 8:
                raise ValueError("NIfTI datatype and bitpix disagree")
            data_bytes = count * dtype.itemsize
            offset = struct.unpack(endian + "f", header[108:112])[0]
            if (not math.isfinite(offset) or offset != int(offset)
                    or not 352 <= offset <= _MAX_BYTES
                    or data_bytes > _MAX_BYTES or offset + data_bytes > _MAX_BYTES):
                raise ValueError("Invalid or unsafe NIfTI data offset/size")
            slope, intercept = struct.unpack(endian + "2f", header[112:120])
            if slope not in (0.0, 1.0) or intercept != 0.0:
                raise ValueError("Only finite unscaled NIfTI values are supported")
            if (header[123] & 7) not in (0, 2):
                raise ValueError("NIfTI spatial units must be millimetres or unspecified")
            if struct.unpack(endian + "h", header[254:256])[0] <= 0:
                raise ValueError("NIfTI requires an explicit sform affine")
            affine = np.eye(4)
            affine[:3] = np.asarray(struct.unpack(endian + "12f", header[280:328])).reshape(3, 4)
            affine = _affine(affine)
            skipped = stream.read(int(offset) - 352)
            if len(skipped) != int(offset) - 352:
                raise ValueError("Truncated NIfTI extension area")
            payload = stream.read(data_bytes)
            if len(payload) != data_bytes:
                raise ValueError("Truncated NIfTI voxel data")
    except (EOFError, gzip.BadGzipFile, struct.error, zlib.error) as error:
        raise ValueError("Malformed NIfTI data: " + str(error)) from None
    array = np.frombuffer(payload, dtype=dtype).reshape(shape, order="F").copy(order="F")
    if not np.isfinite(array).all():
        raise ValueError("NIfTI voxel values must be finite")
    return array, affine


def project_to_frame(array, affine_ras, frame):
    """Nearest-neighbour sample a 3-D array at native DICOM pixel centres.

    ``frame`` uses the existing source-frame contract: ``rows``, ``columns``,
    ``position`` (IPP in LPS mm), ``orientation`` (six IOP direction cosines),
    and ``spacing`` (DICOM PixelSpacing: row, column). Output has shape
    ``(rows, columns)`` and the source array dtype. Outside voxel footprints
    [-0.5, size-0.5) values are zero. Half-voxel ties choose the higher index.
    No transform between independent Frames of Reference is estimated here.
    """
    import numpy as np

    source = np.asarray(array)
    if (source.ndim != 3 or min(source.shape) < 1 or max(source.shape) > _MAX_DIMENSION
            or source.size > _MAX_VOXELS
            or source.dtype.kind not in "biuf" or not np.isfinite(source).all()):
        raise ValueError("A bounded finite numeric 3-D source array is required")
    affine = _affine(affine_ras)
    try:
        rows, columns = frame["rows"], frame["columns"]
        if (type(rows) is not int or type(columns) is not int
                or not 1 <= rows <= _MAX_DIMENSION or not 1 <= columns <= _MAX_DIMENSION
                or rows * columns > _MAX_FRAME_PIXELS):
            raise ValueError("Invalid or unsafe source frame dimensions")
        position, column_step, row_step, _ = _axes(frame)
    except (KeyError, TypeError):
        raise ValueError("Frame requires rows/columns and DICOM position/orientation/spacing") from None
    y, x = np.indices((rows, columns), dtype=np.float64)
    lps = (np.asarray(position)[:, None]
           + np.asarray(column_step)[:, None] * x.ravel()
           + np.asarray(row_step)[:, None] * y.ravel())
    ras = lps * np.array([-1.0, -1.0, 1.0])[:, None]
    coordinates = np.linalg.solve(affine[:3, :3], ras - affine[:3, 3, None])
    valid = np.isfinite(coordinates).all(axis=0)
    valid &= ((coordinates >= -0.5)
              & (coordinates < np.asarray(source.shape)[:, None] - 0.5)).all(axis=0)
    indices = np.floor(coordinates[:, valid] + 0.5).astype(np.intp)
    result = np.zeros(rows * columns, dtype=source.dtype)
    result[valid] = source[tuple(indices)]
    return result.reshape(rows, columns)
