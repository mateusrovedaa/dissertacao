"""Data Compression Algorithms for ViSPAC.

This module implements the compression algorithms used in the ViSPAC edge-fog-cloud
architecture for healthcare IoT vital signs monitoring.

Compression Strategy:
    ViSPAC uses a two-stage compression approach:
    
    1. Lossy Compression (SDT - Swinging Door Trending):
       Applied at the edge to reduce data volume while preserving signal trends.
       Configurable tolerance allows trading off compression ratio vs. accuracy.
    
    2. Lossless Compression (Huffman or LZW):
       Applied to batched data before transmission to further reduce size.
       - Huffman: Better for small payloads (1KB-32KB)
       - LZW: Better for larger payloads (>32KB)

Algorithm References:
    - Bristol, E.H. "Swinging Door Trending: Adaptive Trend Recording?"
      ISA National Conference Proceedings, 1990.
      The SDT algorithm is widely used in SCADA systems and industrial IoT.
    
    - Huffman, D.A. "A Method for the Construction of Minimum-Redundancy Codes."
      Proceedings of the IRE, 40(9):1098-1101, 1952.
    
    - Welch, T.A. "A Technique for High-Performance Data Compression."
      Computer, 17(6):8-19, 1984. (LZW algorithm)

Quality Metrics:
    The Percent Root-mean-square Difference (PRD) is used to evaluate
    lossy compression quality:
    
    PRD = sqrt(sum((original - reconstructed)^2) / sum(original^2)) * 100
    
    Lower PRD indicates better signal fidelity. Typical acceptable values:
    - PRD < 1%: Excellent (suitable for diagnostic purposes)
    - PRD < 5%: Good (suitable for monitoring)
    - PRD < 10%: Acceptable (suitable for trend analysis)

Author: Mateus Roveda
Master's Dissertation - ViSPAC Project
"""

import base64
import heapq
import json
from collections import Counter
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------
# auxiliary functions
# ---------------------------------------------------------------------

def to_b64(b: bytes) -> str:
    """Encode bytes to base64 string for JSON-safe transmission.
    
    Args:
        b: Raw bytes to encode.
    
    Returns:
        str: Base64-encoded string.
    """
    return base64.b64encode(b).decode()


def from_b64(s: str) -> bytes:
    """Decode base64 string back to bytes.
    
    Args:
        s: Base64-encoded string.
    
    Returns:
        bytes: Decoded raw bytes.
    """
    return base64.b64decode(s.encode())

# ---------------------------------------------------------------------
# SDT
# ---------------------------------------------------------------------
class SwingingDoorCompressor:
    """Swinging Door Trending (SDT) lossy compression for time-series data.
    
    SDT is an online data compression algorithm particularly effective for
    slowly-changing sensor data like vital signs. It works by maintaining
    a 'swinging door' formed by lines from an anchor point through tolerance
    bounds (±deviation) of subsequent points.
    
    Algorithm:
        1. Start with first point as anchor
        2. For each new point, compute slopes to upper/lower bounds
        3. Track the narrowest corridor (upper/lower slope bounds)
        4. When new point falls outside corridor, archive previous point
           as new anchor and reset
        5. Also archive if max time gap exceeded (t_max parameter)
    
    Advantages:
        - Linear time complexity O(n)
        - Online algorithm (can process streaming data)
        - Predictable worst-case error (bounded by deviation parameter)
        - Preserves trend information and inflection points

    """
    
    def __init__(self):
        """Initialize the SDT compressor."""
        pass

    def compress(self, time_series: List[Tuple[float, float]], dc: float, t_max_s: int) -> List[Tuple[float, float]]:
        """Compress a time-series using Swinging Door Trending.
        
        Args:
            time_series: List of (timestamp, value) tuples representing the signal.
            dc: Deviation/tolerance parameter (door width). Larger values = more compression.
                For vital signs, typical values:
                - Heart rate: 2-5 bpm
                - SpO2: 1-2 %
            t_max_s: Maximum time gap in seconds before forcing a new anchor point.
                Ensures regular sampling even during stable periods.
        
        Returns:
            List of (timestamp, value) tuples representing compressed signal.
            Always includes first and last points of input.
        
        Example:
            >>> sdt = SwingingDoorCompressor()
            >>> data = [(0, 70), (1, 71), (2, 72), (3, 71), (4, 70)]
            >>> sdt.compress(data, dc=2, t_max_s=60)
            [(0, 70), (4, 70)]  # Only endpoints kept if within tolerance
        """
        if not time_series:
            return []
        archived = [time_series[0]]
        anchor = time_series[0]
        upper, lower = float("inf"), float("-inf")
        t_cap = t_max_s
        for i in range(1, len(time_series)):
            ts, val = time_series[i]
            dt = ts - anchor[0]
            if dt <= 0:
                continue
            if dt > t_cap:
                archived.append(time_series[i-1])
                anchor = time_series[i-1]
                upper, lower = float("inf"), float("-inf")
                dt = ts - anchor[0]
            slope = (val - anchor[1]) / dt
            upper = min(upper, (val + dc - anchor[1]) / dt)
            lower = max(lower, (val - dc - anchor[1]) / dt)
            if not (lower <= slope <= upper):
                archived.append(time_series[i-1])
                anchor = time_series[i-1]
                upper, lower = float("inf"), float("-inf")
        if time_series[-1] not in archived:
            archived.append(time_series[-1])
        return archived

# ---------------------------------------------------------------------
# LZW – text methods return base64; bytes methods operate on raw bytes
# ---------------------------------------------------------------------
class LZW:
    """Lempel-Ziv-Welch (LZW) lossless compression.
    
    LZW is a dictionary-based compression algorithm that builds a table
    of patterns seen in the input. Effective for larger payloads with
    repetitive patterns.
    
    Algorithm:
        Compression:
        1. Initialize dictionary with single-character strings (0-255)
        2. Find longest string W in dictionary matching input
        3. Output dictionary index of W
        4. Add W + next character to dictionary
        5. Repeat from step 2 with remaining input
        
        Decompression:
        1. Initialize dictionary with single-character strings
        2. Read code, output corresponding string
        3. Add previous string + first char of current to dictionary
        4. Repeat
    
    Complexity:
        - Time: O(n) for both compression and decompression
        - Space: O(dictionary size), grows with input patterns

    """
    
    def compress(self, text: str) -> str:
        """Compress text using LZW algorithm.
        
        Args:
            text: Input string to compress.
        
        Returns:
            str: Base64-encoded JSON array of dictionary indices.
        """
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        w = ""
        result: List[int] = []
        for c in text:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = c
        if w:
            result.append(dictionary[w])
        return to_b64(json.dumps(result).encode())

    def decompress(self, b64: str) -> str:
        """Decompress LZW-compressed data.
        
        Args:
            b64: Base64-encoded JSON array of dictionary indices.
        
        Returns:
            str: Original decompressed text.
        """
        data: List[int] = json.loads(from_b64(b64))
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
        if not data:
            return ""
        w = chr(data.pop(0))
        res = [w]
        for k in data:
            entry = dictionary[k] if k in dictionary else w + w[0]
            res.append(entry)
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
            w = entry
        return "".join(res)

    def compress_bytes(self, data: bytes) -> bytes:
        """Compress raw bytes using LZW algorithm.
        
        Operates directly on byte sequences without base64 or JSON overhead.
        Dictionary codes are packed as 4-byte big-endian unsigned integers.
        
        Args:
            data: Raw bytes to compress.
        
        Returns:
            bytes: Compressed byte stream (packed uint32 codes).
        """
        import struct
        dict_size = 256
        dictionary = {bytes([i]): i for i in range(dict_size)}
        w = b""
        result: List[int] = []
        for byte in data:
            b = bytes([byte])
            wc = w + b
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = b
        if w:
            result.append(dictionary[w])
        return struct.pack(f'>{len(result)}I', *result)

    def decompress_bytes(self, data: bytes) -> bytes:
        """Decompress LZW-compressed bytes.
        
        Args:
            data: Compressed byte stream (packed uint32 codes).
        
        Returns:
            bytes: Original decompressed bytes.
        """
        import struct
        if not data:
            return b""
        n = len(data) // 4
        codes = list(struct.unpack(f'>{n}I', data))
        dict_size = 256
        dictionary = {i: bytes([i]) for i in range(dict_size)}
        w = dictionary[codes.pop(0)]
        res = [w]
        for k in codes:
            entry = dictionary[k] if k in dictionary else w + w[0:1]
            res.append(entry)
            dictionary[dict_size] = w + entry[0:1]
            dict_size += 1
            w = entry
        return b"".join(res)

# ---------------------------------------------------------------------
# Huffman – text methods use base64; bytes methods operate on raw bytes
# ---------------------------------------------------------------------
class Huffman:
    """Huffman coding lossless compression.
    
    Huffman coding assigns variable-length codes to symbols based on
    their frequency - more frequent symbols get shorter codes.
    Optimal for small to medium payloads.
    
    Algorithm:
        1. Count frequency of each character/byte
        2. Build binary tree bottom-up:
           - Start with leaf nodes for each symbol
           - Repeatedly merge two lowest-frequency nodes
        3. Assign codes by traversing tree (0=left, 1=right)
        4. Encode input using generated codes
    
    Complexity:
        - Time: O(n log n) for building tree, O(n) for encoding
        - Space: O(k) where k is alphabet size
    
    Output Format (text methods):
        Returns dict with:
        - payload: Base64-encoded compressed bitstream
        - codes: Dict mapping characters to their Huffman codes
        - padding: Number of padding bits added to complete last byte
    
    Output Format (bytes methods):
        Returns a single binary blob containing:
        - 4 bytes: header length (uint32 big-endian)
        - header: msgpack-encoded dict with codes and padding
        - compressed bitstream bytes

    """
    
    def _tree(self, freq: Dict[str, int]):
        """Build Huffman tree and return character-to-code mapping.
        
        Args:
            freq: Dictionary mapping characters to their frequencies.
        
        Returns:
            List of [character, code] pairs sorted by code length.
        """
        heap = [[wt, [ch, ""]] for ch, wt in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for p in lo[1:]:
                p[1] = '0' + p[1]
            for p in hi[1:]:
                p[1] = '1' + p[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

    def compress(self, text: str) -> Dict[str, any]:
        """Compress text using Huffman coding.
        
        Args:
            text: Input string to compress.
        
        Returns:
            Dict with 'payload' (base64 encoded), 'codes' (char -> binary string),
            and 'padding' (bits added for byte alignment).
        """
        if not text:
            return {"payload": "", "codes": {}, "padding": 0}
        freq = Counter(text)
        tree = self._tree(freq)
        codes = {ch: code for ch, code in tree}
        encoded_bits = "".join(codes[ch] for ch in text)
        
        padding = 8 - len(encoded_bits) % 8
        if padding == 8:
            padding = 0
        encoded_bits += '0' * padding
        
        payload_bytes = int(encoded_bits, 2).to_bytes(len(encoded_bits) // 8, 'big')
        return {"payload": to_b64(payload_bytes), "codes": codes, "padding": padding}

    def decompress(self, payload_b64: str, codes: Dict[str, str], padding: int) -> str:
        """Decompress Huffman-coded data.
        
        Args:
            payload_b64: Base64-encoded compressed bitstream.
            codes: Dictionary mapping characters to their Huffman codes.
            padding: Number of padding bits to strip from end.
        
        Returns:
            str: Original decompressed text.
        """
        if not payload_b64:
            return ""
        rev = {v: k for k, v in codes.items()}
        
        payload_bytes = from_b64(payload_b64)
        bitstr = bin(int.from_bytes(payload_bytes, 'big'))[2:].zfill(len(payload_bytes) * 8)
        if padding > 0:
            bitstr = bitstr[:-padding]  # Remove padding
        
        curr = ""
        out = []
        for bit in bitstr:
            curr += bit
            if curr in rev:
                out.append(rev[curr])
                curr = ""
        return "".join(out)

    def compress_bytes(self, data: bytes) -> bytes:
        """Compress raw bytes using Huffman coding.
        
        Packs the codebook and compressed bitstream into a single binary blob:
        [4-byte header_len][msgpack header (codes + padding)][compressed bits]
        
        Args:
            data: Raw bytes to compress.
        
        Returns:
            bytes: Single binary blob with embedded codebook and compressed data.
        """
        import struct, msgpack
        if not data:
            return b""
        freq = Counter(data)
        # Build tree using byte values (int keys)
        heap = [[wt, [ch, ""]] for ch, wt in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for p in lo[1:]:
                p[1] = '0' + p[1]
            for p in hi[1:]:
                p[1] = '1' + p[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        tree = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
        codes = {ch: code for ch, code in tree}
        
        encoded_bits = "".join(codes[b] for b in data)
        padding = 8 - len(encoded_bits) % 8
        if padding == 8:
            padding = 0
        encoded_bits += '0' * padding
        
        compressed = int(encoded_bits, 2).to_bytes(len(encoded_bits) // 8, 'big')
        # Pack header: codes dict (int->str) + padding
        header = msgpack.packb({"codes": codes, "padding": padding})
        return struct.pack('>I', len(header)) + header + compressed

    def decompress_bytes(self, data: bytes) -> bytes:
        """Decompress Huffman-coded binary blob.
        
        Args:
            data: Binary blob from compress_bytes().
        
        Returns:
            bytes: Original decompressed bytes.
        """
        import struct, msgpack
        if not data:
            return b""
        header_len = struct.unpack('>I', data[:4])[0]
        header = msgpack.unpackb(data[4:4+header_len], strict_map_key=False)
        compressed = data[4+header_len:]
        
        codes = header[b"codes"] if b"codes" in header else header["codes"]
        padding = header[b"padding"] if b"padding" in header else header["padding"]
        
        # Build reverse lookup: bitcode string -> byte value
        rev = {}
        for k, v in codes.items():
            byte_val = k if isinstance(k, int) else ord(k)
            code_str = v if isinstance(v, str) else v.decode()
            rev[code_str] = byte_val
        
        bitstr = bin(int.from_bytes(compressed, 'big'))[2:].zfill(len(compressed) * 8)
        if padding > 0:
            bitstr = bitstr[:-padding]
        
        curr = ""
        out = bytearray()
        for bit in bitstr:
            curr += bit
            if curr in rev:
                out.append(rev[curr])
                curr = ""
        return bytes(out)