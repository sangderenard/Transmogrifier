class _RawSpan(bytearray):
    """A zero‑copy view (slice) tied to a BitBitBuffer plane."""
    __slots__ = ("_bitbit_cap", "_origin", "_offset")
    def __init__(self, backing, start_bit=0, length_bits=None, readonly=False):
        pass
    def __new__(cls, backing, offset, length):
        # backing: original bytearray; offset, length in bits
        view = super().__new__(cls,
            backing[offset // 8 : (offset + length + 7) // 8]
        )
        view._origin = backing
        view._offset = offset
        view._bitbit_cap = True
        return view

class BitBitItem:
    def __init__(self, buffer=None, mask_index=None, length=None, cast=None, padding=0, padding_mask=None):
        self.id = id(self)
        self.buffer = buffer
        self.mask_index = mask_index
        self.padding = padding
        self.padding_mask = padding_mask or 0
        self.padded_length = length
        self.useful_length = length - padding
        self.cast = cast or int
        if self.mask_index is not None:
            self.data_index = self.buffer.bitsforbits * self.mask_index
        else:
            self.data_index = None
    def __len__(self):
        return self.useful_length

    def padded_length(self):
        return self.padded_length

    def set_index(self, index):
        self.mask_index = index
        self.data_index = self.buffer.bitsforbits * self.mask_index

    def __setitem__(self, index, value):
        if index == 'mask':
            self.buffer[self.mask_index:self.mask_index + self.useful_length] = value
        elif index == 'data':
            self.buffer._data_access[self.mask_index:self.mask_index + self.useful_length] = value
        else:
            raise KeyError(f"Invalid index: {index}")

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.useful_length
            if stop < start:
                raise IndexError("Slice stop must be >= start")
            length = stop - start
            return BitBitSlice(self.buffer, self.mask_index + start, length, reversed=index.step < 0)
        elif isinstance(index, int):
            if not (0 <= index < self.useful_length):
                raise IndexError("Bit index out of range")
            # Return a BitBitItem that will read/write this mask bit
            return BitBitItem(buffer=self.buffer, mask_index=self.mask_index + index, length=1, cast=self.cast)
        if index == 'mask':
            return self.buffer[self.mask_index:self.mask_index + self.useful_length]
        elif index == 'data':
            self.buffer._data_access.caster = self.cast
            return self.buffer._data_access[self.mask_index:self.mask_index + self.useful_length]
        else:
            raise KeyError(f"Invalid index: {index}")

    def __int__(self):
        # Return raw mask bits as integer
        if self.useful_length == 1:
            idx = self.mask_index
            byte_i = idx // 8
            bit_off = 7 - (idx % 8)
            return (self.buffer.mask[byte_i] >> bit_off) & 1
        else:
            raw = self.buffer.extract_bit_region(self.buffer.mask,
                                                self.mask_index,
                                                self.useful_length)
            return int.from_bytes(raw, byteorder='big')

    def __repr__(self):
        return f"BitBitItem(len={self.useful_length}, idx={self.mask_index})"

class BitBitSlice(BitBitItem):
    """
    Immutable view on an *aligned* bit‑range.
    """
    __slots__ = ("reversed",)

    def __init__(self, buffer, start_bit, length, reversed=False):
        stride = buffer.bitsforbits
        padded = buffer.intceil(length, stride)
        padding = padded - length
        super().__init__(buffer, start_bit, padded, cast=bytearray, padding=padding)
        self.reversed = reversed

    def __bytes__(self):
        # Don’t read past the end of the mask plane
        avail_len = max(0, self.buffer.mask_size - self.mask_index)
        avail_len = min(avail_len, len(self))
        raw = self.buffer.extract_bit_region(
            self.buffer.mask,
            self.mask_index,
            avail_len
        )
        return bytes(raw)


    def __repr__(self):
        b = bytes(self)
        bit_str = ''.join(f'{byte:08b}' for byte in b)
        return f"BitBitSlice({bit_str})"

class BitBitBufferDataAccess:
    def __init__(self, buffer, caster=int):
        self.buffer = buffer
        self.caster = caster

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.buffer.mask_size
            length = stop - start
            data_index = self.buffer.bitsforbits * start
            total_bits = self.buffer.bitsforbits * length
        elif isinstance(index, int):
            start = index
            length = 1
            data_index = self.buffer.bitsforbits * start
            total_bits = self.buffer.bitsforbits
        else:
            raise TypeError("Invalid index type for data access")

        raw_bytes = self.buffer.extract_bit_region(self.buffer.data, data_index, total_bits)
        
        # Built-in cast logic
        caster = self.caster
        if caster is int:
            return int.from_bytes(raw_bytes, byteorder='big')
        elif caster is float:
            return int.from_bytes(raw_bytes, byteorder='big') / (2**total_bits - 1)
        elif caster is bytes:
            return bytes(raw_bytes)
        elif caster is bytearray:
            return raw_bytes
        elif caster is tuple:
            return tuple(raw_bytes)
        else:
            return caster(raw_bytes)


    def __setitem__(self, index, value):
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.buffer.mask_size
            length = stop - start
            data_index = self.buffer.bitsforbits * start
            total_bits = self.buffer.bitsforbits * length

        elif isinstance(index, int):
            start = index
            length = 1
            data_index = self.buffer.bitsforbits * index
            total_bits = self.buffer.bitsforbits

        else:
            raise TypeError("Invalid index type for data access")

        # If given an int, convert to padded bit-packed buffer
        if isinstance(value, int):
            value = self.buffer.build_fill_buffer(value, total_bits)

        self.buffer.write_bit_region(self.buffer.data, data_index, value, total_bits)


class BitBitBuffer:
    def __init__(self, data_size=None, mask_size=None, bitsforbits=None):
        self.bitsforbits = bitsforbits or 8
        if data_size is None and mask_size is None:
            raise ValueError("At least one of data_size or mask_size must be specified")
        self.data_size = data_size if data_size is not None else mask_size * self.bitsforbits
        self.mask_size = mask_size if mask_size is not None else data_size // self.bitsforbits
        self.data = bytearray(self.bittobyte(self.data_size))
        self.mask = bytearray(self.bittobyte(self.mask_size))
        self._data_access = BitBitBufferDataAccess(self)

        # ---------------------------------------------------------------
        # internal capability token
        # ---------------------------------------------------------------
        self._bitbit_internal = object()

    # ===============================================================
    # INTERNAL FAST‑PATH HELPERS
    # ===============================================================
    def _auth_view(self, backing: bytearray, readonly: bool):
        """Return a view carrying the _bitbit_cap flag."""
        return _RawSpan(backing, readonly)

    def _span(self, plane: str, start_bit: int, length_bits: int, *, readonly: bool = False):
        """
        INTERNAL: zero‑copy slice of mask/data with capability flag.
        """
        backing = self.mask if plane == "mask" else self.data
        return _RawSpan(backing, start_bit, length_bits)
    
    def __len__(self):
        return self.mask_size
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop  = self.mask_size if index.stop is None else index.stop
            step  = index.step or 1
            rev   = (step < 0)
            # only forbid forward slices with stop < start
            if not rev and stop < start:
                raise IndexError("Slice stop must be >= start")
            if rev and index.start is not None and index.stop is not None:
                # invert bounds for reverse iteration
                start, stop = index.stop + 1, index.start + 1
            length = stop - start
            return BitBitSlice(self, start, length, reversed=rev)

        elif isinstance(index, int):
            if not (0 <= index < self.mask_size):
                raise IndexError("Bit index out of range")
            # Return a BitBitItem that will read/write this mask bit
            return BitBitItem(buffer=self, mask_index=index, length=1, cast=int)
    def __setitem__(self, index, value):
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.mask_size
            if stop < start:
                raise IndexError("Slice stop must be >= start")
            length = stop - start
            # Allow list of items (or ints)
            if isinstance(value, list):
                # Convert BitBitItem objects or values to bits (0 or 1)
                bits = [int(v) if isinstance(v, BitBitItem) else int(v) for v in value]
                buf = bytearray((length + 7) // 8)
                for i, bit in enumerate(bits):
                    if bit:
                        byte_i = i // 8
                        bit_offset = 7 - (i % 8)
                        buf[byte_i] |= (1 << bit_offset)
                fill_buf = buf
            elif isinstance(value, BitBitSlice):
                fill_buf = bytes(value)     
            elif isinstance(value, int):
                fill_buf = self.build_fill_buffer(value, length)
            elif isinstance(value, (bytes, bytearray)):
                if len(value) * 8 < length:
                    raise ValueError("Not enough data to fill slice")
                fill_buf = value
            else:
                raise TypeError("Slice assignment must be int, list, or bytearray")
            self.write_bit_region(self.mask, start, fill_buf, length)
        elif isinstance(index, int):
            if not (0 <= index < self.mask_size):
                raise IndexError("Bit index out of range")
            if isinstance(value, BitBitItem):
                value = int(value)
            byte_index = index // 8
            bit_offset = 7 - (index % 8)
            if value:
                self.mask[byte_index] |= (1 << bit_offset)
            else:
                self.mask[byte_index] &= ~(1 << bit_offset)
        else:
            raise TypeError("Index must be int or slice")
    
    def __repr__(self):
        width = 80
        ramp = " .*#@"
        bits = self.to_bitstring()
        chars = []
        for i in range(width):
            bit = bits[i * len(bits) // width]
            if bit == '1':
                if len(chars) < i + 1:
                    chars.append(0) 
                chars[i] += 1
        max = max(chars) if chars else 1
        bar = ''.join(ramp[min(int(c * len(ramp) / max), len(ramp) - 1)] for c in chars)
        return f"BitBitBuffer({self.bittobyte(self.data_size)} bytes catalogued in {self.mask_size} bits\n {bar}"

    def __iter__(self):
        for i in range(self.mask_size):
            yield self[i]

    def move(self, src, dst, length):
        self[dst:dst + length] = self[src:src + length]

    def swap(self, src, dst, length):
        """Swap a region of bits from src to dst."""
        src_bits = self[src:src + length]
        dst_bits = self[dst:dst + length]
        self[dst:dst + length] = src_bits
        self[src:src + length] = dst_bits
    # -- low‑level primitive --------------------------------------------------
    def _insert_bits(self, mask_off: int, mask_bits: int) -> None:
        """
        Insert `mask_bits` blank bits at mask‑index `mask_off`.
        The data plane is expanded automatically (bitsforbits × mask_bits),
        and existing payload is shifted right.

        NOTE: *nothing* is written to the newly‑created span; callers decide
        whether it stays 0 or gets stamped later.
        """
        assert 0 <= mask_off <= self.mask_size + 1, f"Invalid mask offset {mask_off} for mask size {self.mask_size}"

        # -------- mask plane --------------------------------------------------
        new_mask_size  = self.mask_size + mask_bits
        new_mask_bytes = self.bittobyte(new_mask_size)
        new_mask = bytearray(new_mask_bytes)
        mask_view = _RawSpan(new_mask, 0, new_mask_size)
        # copy left block
        if mask_off:
            left = self.extract_bit_region(self.mask, 0, mask_off)
            self.write_bit_region(mask_view, mask_bits, left, mask_off)
        # copy right block (shifted)
        right_len = self.mask_size - mask_off
        if right_len > 0:
            right = self.extract_bit_region(self.mask, mask_off, right_len)
            self.write_bit_region(mask_view, mask_off + mask_bits, right, right_len)

        print(f"Expanding mask from {self.mask_size} to {new_mask_size} bits at offset {mask_off}")

        self.mask       = new_mask
        self.mask_size  = new_mask_size

        # -------- data plane --------------------------------------------------
        data_off       = mask_off * self.bitsforbits
        data_bits      = mask_bits * self.bitsforbits
        new_data_size  = self.data_size + data_bits
        new_data_bytes = self.bittobyte(new_data_size)
        new_data = bytearray(new_data_bytes)
        # internal view for the data plane
        data_view = _RawSpan(new_data, 0, data_off)

        if data_off:
            left = self.extract_bit_region(self.data, 0, data_off)
            self.write_bit_region(data_view, 0, left, data_off)

        right_len = self.data_size - data_off
        if right_len:
            right = self.extract_bit_region(self.data, data_off, right_len)
            self.write_bit_region(data_view, data_off + data_bits, right, right_len)
            
        self.data      = new_data
        self.data_size = new_data_size
    # ------------------------------------------------------------------------


    # -- public façade --------------------------------------------------------
    def expand(self, events, cells=None, proposals=None):
        """
        `events`  : iterable[(mask_offset, mask_bits_to_insert)] sorted asc.
        `cells`   : optional list of Cell objects to auto‑shift .left/.right .
        """
        shift = 0

        for label, off, sz in events:
            assert off <= self.mask_size, \
                f"Invalid mask offset {off} for mask size {self.mask_size}"
            adj_off = off + shift    
            previous_mask_size = self.mask_size
            assert 0 <= adj_off <= self.mask_size, \
                f"Invalid adjusted mask offset {adj_off} for mask size {self.mask_size}"
            self._insert_bits(adj_off, sz)
            assert self.mask_size == previous_mask_size + sz, \
                f"Mask size mismatch after insertion: {self.mask_size} != {previous_mask_size} + {sz}"
            shift += sz
            if cells:
                for c in cells:
                    if c.left >= adj_off:
                        c.left  += sz
                        c.leftmost += sz
                        c.right += sz
                        c.rightmost += sz
                    elif c.left < adj_off < c.right:   # insertion *inside* cell
                        if c.leftmost < adj_off < c.rightmost:
                            c.rightmost += sz
                        if c.leftmost > adj_off:
                            c.leftmost += sz
                            c.rightmost += sz
                        c.right += sz
            
    def _count_runs(self):
        """Count consecutive runs of a specific value in the data."""
        count = [None, 0]
        runs = []
        last_bit = None
        for bit in self:
            if bit == last_bit:
                count[1] += 1
            else:
                last_bit = bit
                if count[1] > 0:
                    runs.append(count)
                count = [last_bit, 1]
        if count > 0:
            runs.append(count)
        return runs

    def tuplepattern(self, src, length, direction='left'):
        if direction == 'left' or direction == 'bi':
            reversed = self[src:src + length][::-1]
            left_pattern = self._count_runs(reversed)

        if direction == 'right' or direction == 'bi':
            right_pattern = self._count_runs(self[src:src + length])
                
        if direction == 'left':
            return left_pattern
        elif direction == 'right':
            return right_pattern
        elif direction == 'bi':
            return left_pattern, right_pattern
            
    def to_bitstring(self):
        # Return a string representing the data bits 
        return ''.join(str(bit) for bit in self)

    def bittobyte(self, bits):
        return self.intceil(bits) // 8

    @staticmethod
    def _intceil(val, base=8):
        """Return the smallest multiple of `base` that is >= `val`."""
        return (val + base - 1) // base * base

    def intceil(self, val, base=8):
        return self._intceil(val, base)

    def build_fill_buffer(self, fill_value: int, length_bits: int) -> bytearray:
        nbytes = (length_bits + 7) // 8
        shift = nbytes * 8 - length_bits
        if fill_value == 1:
            fill_value = (2**length_bits - 1)
        buf = bytearray((fill_value << shift).to_bytes(nbytes, byteorder='big'))
        return self.mask_padding_bits(buf, length_bits)

    # NEW: Helper method to wrap raw input if not already a metadata object.
    def _ensure_metadata(self, raw, default_mask_index=0):
        if hasattr(raw, "mask_index"):
            return raw
        else:
            raise TypeError("Stamping requires a BitBitItem or BitBitSlice from self[...]")
    
    def stamp(self, raw, indices, default_stride, default_value=1):
        # Enforce metadata wrapping at the very beginning
        raw = self._ensure_metadata(raw, getattr(raw, "mask_index", 0))
        for entry in indices:
            if isinstance(entry, (tuple, list)):
                if len(entry) == 2:
                    gap, stride = entry
                    value = default_value
                elif len(entry) == 3:
                    gap, stride, value = entry
                else:
                    raise ValueError("Invalid entry format")
            else:
                gap = entry
                stride = default_stride
                value = default_value
            fill_buf = self.build_fill_buffer(value, stride)
            self.write_bit_region(raw, gap, fill_buf, stride)
        return raw

    def write_bit_region(self, target, start_bit: int, buf, length_bits: int) -> None:
        # Accept internal metadata objects or direct mask/data
        from bitbitbuffer import BitBitItem
        if isinstance(target, BitBitItem):
            backing = target.buffer.mask
            base = target.mask_index
        elif getattr(target, '_bitbit_cap', False):
            backing = target._origin if hasattr(target, '_origin') else target
            base = getattr(target, '_offset', 0)
        elif target is self.mask or target is self.data:
            backing = target
            base = 0
        else:
            raise TypeError("Target for writing must be a BitBitItem or BitBitSlice instance")

        for i in range(length_bits):
            src_byte = buf[i // 8]
            src_bit_val = (src_byte >> (7 - (i % 8))) & 1
            dest_bit = base + start_bit + i
            dest_byte_i = dest_bit // 8
            dest_bit_offset = 7 - (dest_bit % 8)
            if src_bit_val:
                backing[dest_byte_i] |= (1 << dest_bit_offset)
            else:
                backing[dest_byte_i] &= ~(1 << dest_bit_offset)

    def extract_bit_region(self, data, start_bit: int, length: int) -> bytearray:
        # Accept raw storage or capability‑marked views.
        if getattr(data, "_bitbit_cap", False):
            backing   = data._origin
            start_bit = data._offset + start_bit
            data      = backing
        elif not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Extraction source must be bytes‑like or an internal view")
        if not isinstance(data, (bytes, bytearray)):
            data = bytes(data)
        out = bytearray((length + 7) // 8)
        for i in range(length):
            src_bit = (data[(start_bit + i) // 8] >> (7 - ((start_bit + i) % 8))) & 1
            out[i // 8] |= src_bit << (7 - (i % 8))
        return out

    def mask_padding_bits(self, buf: bytearray, length_bits: int) -> bytearray:
        total_bits = len(buf) * 8
        pad_bits = total_bits - length_bits
        if pad_bits <= 0:
            return buf
        for bit_index in range(length_bits, total_bits):
            byte_i = bit_index // 8
            bit_offset = 7 - (bit_index % 8)
            buf[byte_i] &= ~(1 << bit_offset)
        return buf


# test_bitbitbuffer.py
import pytest
from bitbitbuffer import BitBitBuffer, BitBitItem, BitBitSlice   # adjust import path as needed


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _alt_pattern(n):
    """00000000 -> 10101010 pattern for n bits."""
    out = bytearray((n + 7)//8)
    for i in range(n):
        if i % 2:
            byte = i // 8
            bit  = 7 - (i % 8)
            out[byte] |= 1 << bit
    return out


# ---------------------------------------------------------------------------
# 1 – construction & basic length
# ---------------------------------------------------------------------------
def test_construction_and_lengths():
    buf = BitBitBuffer(mask_size=13, bitsforbits=4)
    assert len(buf) == 13
    assert buf.data_size == 13 * 4


# ---------------------------------------------------------------------------
# 2 – single‑BitBitItem round‑trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("idx", [0, 6, 12])
def test_single_bit_item(idx):
    buf = BitBitBuffer(mask_size=13, bitsforbits=8)
    item = buf[idx]          # BitBitItem
    item['mask'] = 1
    assert int(buf[idx]) == 1
    item['data'] = 0xAA
    assert buf._data_access[idx] == 0xAA


# ---------------------------------------------------------------------------
# 3 – slice returns BitBitSlice & preserves padding
# ---------------------------------------------------------------------------
def test_slice_padding():
    stride = 6
    buf    = BitBitBuffer(mask_size=20, bitsforbits=stride)
    view   = buf[3:11]                       # 8 payload bits
    assert isinstance(view, BitBitSlice)
    assert view.padding == stride - 8 % stride


# ---------------------------------------------------------------------------
# 4 – reverse slice order
# ---------------------------------------------------------------------------
def test_reverse_slice():
    buf = BitBitBuffer(mask_size=16, bitsforbits=4)
    # set four consecutive bits so the pattern is visible
    buf[4:8] = 0b1111
    fwd = bytes(buf[4:8])           # normal
    rev = bytes(buf[7:3:-1])        # reverse
    # padding is in front for rev; stripping gives same useful bits
    assert fwd.rstrip(b'\x00') == rev.lstrip(b'\x00')


# ---------------------------------------------------------------------------
# 5 – endianness: big‑endian bit order guaranteed
# ---------------------------------------------------------------------------
def test_endian_guarantee():
    buf = BitBitBuffer(mask_size=8, bitsforbits=8)
    buf[1] = 1              # set bit 1 (second bit)
    assert bytes(buf[0:8]) == b'\x40'   # 0100 0000


# ---------------------------------------------------------------------------
# 6 – mask/data coherence through BitBitItem list write
# ---------------------------------------------------------------------------
def test_bulk_slice_write():
    buf  = BitBitBuffer(mask_size=16, bitsforbits=8)
    bits = [BitBitItem(buffer=buf, mask_index=i, length=1) for i in range(8)]
    for i, b in enumerate(bits):
        b['mask'] = (i % 2)
    buf[0:8] = bits                    # write back
    expect = _alt_pattern(8)
    assert bytes(buf[0:8]) == expect


# ---------------------------------------------------------------------------
# 7 – build_fill_buffer pads correctly
# ---------------------------------------------------------------------------
def test_build_fill_buffer():
    buf   = BitBitBuffer(mask_size=1, bitsforbits=5)
    fill  = buf.build_fill_buffer(1, 5)   # exactly one stride
    assert len(fill) == 1                 # 5 bits -> 1 byte


# ---------------------------------------------------------------------------
# 8 – stamp fills gaps at stride
# ---------------------------------------------------------------------------
def test_stamp_and_extract():
    stride = 3
    buf    = BitBitBuffer(mask_size=9, bitsforbits=stride)
    view   = buf[0:stride]               # first triple‑bit slice
    buf.stamp(view, [0], stride)         # set to 1s
    assert bytes(buf[0:stride]) == b'\xE0'   # 111xxxxx


# ---------------------------------------------------------------------------
# 9 – _low‑level insert_bits shifts mask & data equally
# ---------------------------------------------------------------------------
def test_insert_bits_consistency():
    buf = BitBitBuffer(mask_size=8, bitsforbits=2)
    buf[0] = 1
    buf._insert_bits(4, 2)               # insert in the middle
    assert len(buf) == 10
    # original first bit still set, but shifted right by 2
    assert int(buf[2]) == 1


# ---------------------------------------------------------------------------
# 10 – expand() auto‑shifts mock “cells”
# ---------------------------------------------------------------------------

def test_expand_shifts_cells():
    class Dummy:
        def __init__(self, left, right):
            self.left = left
            self.right = right
    cells = [Dummy(0, 4), Dummy(4, 8)]
    buf = BitBitBuffer(mask_size=8, bitsforbits=4)
    buf.expand([(2,2)], cells=cells)
    assert cells[0].right - cells[0].left == 6
    assert cells[1].left == 6


# ---------------------------------------------------------------------------
# 11 – capability flag prevents external mutation
# ---------------------------------------------------------------------------
def test_capability_flag():
    buf  = BitBitBuffer(mask_size=8, bitsforbits=4)
    span = buf._span("mask", 0, 4)            # internal view (allowed)
    buf.write_bit_region(span, 0, b'\xF0', 4) # should not raise
    rogue = bytearray(b'\x00')
    with pytest.raises(TypeError):
        buf.write_bit_region(rogue, 0, b'\xF0', 4)


# ---------------------------------------------------------------------------
# pytest entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__])
    pytest.main([__file__])
