# compressors.py
import json
import heapq
from collections import defaultdict, Counter

class SwingingDoorCompressor:
    """
    Implementa o algoritmo de compressão Swinging Door Trending (SDT).
    Este algoritmo é 'lossy' (com perdas) e é eficaz para comprimir
    séries temporais, preservando os picos e vales importantes.
    """
    def compress(self, time_series, dc_deviation, t_sdt_max_interval):
        if not time_series:
            return []

        archived_points = [time_series[0]]
        anchor_point = time_series[0]
        
        upper_slope = float('inf')
        lower_slope = float('-inf')

        for i in range(1, len(time_series)):
            current_point = time_series[i]
            time_delta = current_point[0] - anchor_point[0]
            
            if time_delta <= 0:
                continue

            if time_delta > t_sdt_max_interval:
                if time_series[i-1] not in archived_points:
                    archived_points.append(time_series[i-1])
                
                anchor_point = time_series[i-1]
                upper_slope = float('inf')
                lower_slope = float('-inf')
                time_delta = current_point[0] - anchor_point[0]
                if time_delta <= 0: continue

            slope = (current_point[1] - anchor_point[1]) / time_delta
            
            new_upper_slope = (current_point[1] + dc_deviation - anchor_point[1]) / time_delta
            new_lower_slope = (current_point[1] - dc_deviation - anchor_point[1]) / time_delta

            upper_slope = min(upper_slope, new_upper_slope)
            lower_slope = max(lower_slope, new_lower_slope)

            if slope > upper_slope or slope < lower_slope:
                if time_series[i-1] not in archived_points:
                    archived_points.append(time_series[i-1])
                anchor_point = time_series[i-1]
                upper_slope = float('inf')
                lower_slope = float('-inf')

        if time_series[-1] not in archived_points:
            archived_points.append(time_series[-1])

        return archived_points

class LZW:
    """Implementação do algoritmo de compressão LZW."""
    def compress(self, uncompressed_string: str) -> list[int]:
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        
        w = ""
        result = []
        for c in uncompressed_string:
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
        return result

    def decompress(self, compressed_data: list[int]) -> str:
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
        
        result = []
        w = chr(compressed_data.pop(0))
        result.append(w)
        
        for k in compressed_data:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + w[0]
            else:
                raise ValueError(f"Bad compressed k: {k}")
            
            result.append(entry)
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
            w = entry
            
        return "".join(result)

class Huffman:
    """Implementação do algoritmo de compressão Huffman."""
    def _build_tree(self, frequencies):
        heap = [[weight, [char, ""]] for char, weight in frequencies.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

    def compress(self, uncompressed_string: str) -> dict:
        frequencies = Counter(uncompressed_string)
        huffman_tree = self._build_tree(frequencies)
        codes = {char: code for char, code in huffman_tree}
        
        encoded_string = "".join(codes[char] for char in uncompressed_string)
        
        return {"payload": encoded_string, "codes": codes}

    def decompress(self, encoded_string: str, codes: dict) -> str:
        reverse_codes = {v: k for k, v in codes.items()}
        current_code = ""
        decoded_string = []
        
        for bit in encoded_string:
            current_code += bit
            if current_code in reverse_codes:
                decoded_string.append(reverse_codes[current_code])
                current_code = ""
                
        return "".join(decoded_string)
