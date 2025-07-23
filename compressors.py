import base64, heapq, json
from collections import Counter
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode()

def from_b64(s: str) -> bytes:
    return base64.b64decode(s.encode())

# ---------------------------------------------------------------------
# SDT – unchanged (accepts per‑sample tuple)
# ---------------------------------------------------------------------
class SwingingDoorCompressor:
    def __init__(self):
        pass

    def compress(self, time_series: List[Tuple[float, float]], dc: float, t_max_s: int):
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
# LZW – returns base64 string
# ---------------------------------------------------------------------
class LZW:
    def compress(self, text: str) -> str:
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        w = ""; result: List[int] = []
        for c in text:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size; dict_size += 1
                w = c
        if w:
            result.append(dictionary[w])
        return to_b64(json.dumps(result).encode())

    def decompress(self, b64: str) -> str:
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
            dictionary[dict_size] = w + entry[0]; dict_size += 1
            w = entry
        return "".join(res)

# ---------------------------------------------------------------------
# Huffman – returns {payload, codes} with payload base64
# ---------------------------------------------------------------------
class Huffman:
    def _tree(self, freq: Dict[str, int]):
        heap = [[wt, [ch, ""]] for ch, wt in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for p in lo[1:]: p[1] = '0' + p[1]
            for p in hi[1:]: p[1] = '1' + p[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

    def compress(self, text: str) -> Dict[str, any]:
        if not text:
            return {"payload": "", "codes": {}, "padding": 0}
        freq = Counter(text)
        tree = self._tree(freq)
        codes = {ch: code for ch, code in tree}
        encoded_bits = "".join(codes[ch] for ch in text)
        
        padding = 8 - len(encoded_bits) % 8
        if padding == 8: padding = 0
        encoded_bits += '0' * padding
        
        payload_bytes = int(encoded_bits, 2).to_bytes(len(encoded_bits)//8, 'big')
        return {"payload": to_b64(payload_bytes), "codes": codes, "padding": padding}

    def decompress(self, payload_b64: str, codes: Dict[str, str], padding: int) -> str:
        if not payload_b64:
            return ""
        rev = {v: k for k, v in codes.items()}
        
        payload_bytes = from_b64(payload_b64)
        bitstr = bin(int.from_bytes(payload_bytes, 'big'))[2:].zfill(len(payload_bytes) * 8)
        if padding > 0:
            bitstr = bitstr[:-padding] # Remove o padding
        
        curr = ""; out = []
        for bit in bitstr:
            curr += bit
            if curr in rev:
                out.append(rev[curr]); curr = ""
        return "".join(out)