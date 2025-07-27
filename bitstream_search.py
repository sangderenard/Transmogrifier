from typing import List, Tuple, Set

# Assumes BitBitSlice implements __getitem__(int) -> int and exposes:
# - bit_length: int
# - padding: int

class BitStreamSearch:
    """
    A stride-aware, BitBitSlice-native bit stream search utility.
    Bit-level logic is strictly centralized to BitBit; all access is via bit index only.
    """

    @staticmethod
    def count_runs(bitslice) -> List[Tuple[int, int]]:
        """
        Run-length encode a BitBitSlice using bit-index access only.
        """
        pattern = []
        last_bit = None
        count = 0
        for i in range(len(bitslice)):
            bit = int(bitslice[i])
            if bit == last_bit:
                count += 1
            else:
                if last_bit is not None:
                    pattern.append((last_bit, count))
                last_bit = bit
                count = 1
        if count > 0:
            pattern.append((last_bit, count))
        return pattern

    @staticmethod
    def find_aligned_zero_runs(pattern: List[Tuple[int, int]], stride: int) -> Set[int]:
        """
        Identify stride-aligned zero-run starts using run-length pattern.
        All returned offsets are local bit indices within the slice.
        """
        offsets = set()
        cursor = 0
        for bit, count in pattern:
            if bit == 0:
                for i in range(0, count, stride):
                    pos = cursor + i
                    if i + stride <= count and pos % stride == 0:
                        offsets.add(pos)
            cursor += count
        return offsets

    @classmethod
    def detect_stride_gaps(cls, bitslice, stride: int) -> Tuple[List[Tuple[int, int]], Set[int]]:
        """
        High-level entry point: returns run-length pattern and set of stride-aligned 0-gaps.
        """
        pattern = cls.count_runs(bitslice)
        gaps = cls.find_aligned_zero_runs(pattern, stride)
        return pattern, gaps
