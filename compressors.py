# compressors.py

import json
import heapq
from collections import defaultdict

class LZW:
    """
    Classe para compressão/descompressão LZW em memória.
    Lógica adaptada para operar com strings e retornar/receber listas de inteiros.
    """
    def compress(self, data_str: str) -> list[int]:
        data_bytes = data_str.encode('utf-8')
        dictionary_size = 256
        dictionary = {bytes([i]): i for i in range(dictionary_size)}
        
        result = []
        p = b""
        for c_byte in data_bytes:
            c = bytes([c_byte])
            pc = p + c
            if pc in dictionary:
                p = pc
            else:
                result.append(dictionary[p])
                dictionary[pc] = dictionary_size
                dictionary_size += 1
                p = c
        if p:
            result.append(dictionary[p])
        return result

    def decompress(self, data_codes: list[int]) -> str:
        dictionary_size = 256
        dictionary = {i: bytes([i]) for i in range(dictionary_size)}
        
        if not data_codes:
            return ""
            
        p = dictionary[data_codes.pop(0)]
        result_bytes = p
        
        for code in data_codes:
            if code in dictionary:
                entry = dictionary[code]
            elif code == dictionary_size:
                entry = p + p[:1]
            else:
                raise ValueError(f'Erro na descompressão LZW: código {code} inválido')
            
            result_bytes += entry
            dictionary[dictionary_size] = p + entry[:1]
            dictionary_size += 1
            p = entry
            
        return result_bytes.decode('utf-8')


class Huffman:
    """
    Classe para compressão/descompressão Huffman, adequada para uso como biblioteca.
    """
    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None
        
        def __lt__(self, other):
            return self.freq < other.freq

    def _make_frequency_dict(self, text):
        return defaultdict(int, {char: text.count(char) for char in set(text)})

    def _make_heap(self, frequency):
        return [self.HeapNode(key, frequency[key]) for key in frequency]

    def _merge_nodes(self, heap):
        heapq.heapify(heap)
        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(heap, merged)
        return heap[0]

    def _make_codes_helper(self, root, current_code, codes_dict):
        if root is None:
            return
        if root.char is not None:
            codes_dict[root.char] = current_code
            return
        self._make_codes_helper(root.left, current_code + "0", codes_dict)
        self._make_codes_helper(root.right, current_code + "1", codes_dict)

    def _make_codes(self, root):
        codes_dict = {}
        self._make_codes_helper(root, "", codes_dict)
        return codes_dict

    def compress(self, text: str):
        frequency = self._make_frequency_dict(text)
        heap = self._make_heap(frequency)
        root = self._merge_nodes(heap)
        codes = self._make_codes(root)
        encoded_text = "".join(codes[character] for character in text)
        return encoded_text, codes

    def decompress(self, encoded_text: str, codes: dict):
        reverse_mapping = {v: k for k, v in codes.items()}
        current_code = ""
        decoded_text = ""
        for bit in encoded_text:
            current_code += bit
            if current_code in reverse_mapping:
                decoded_text += reverse_mapping[current_code]
                current_code = ""
        return decoded_text


class SwingingDoor:
    """
    Versão robusta e simplificada do SwingingDoor para tempo real.
    Um ponto só é significativo se variar em relação ao último ponto ARQUIVADO.
    """
    def __init__(self, deviation):
        self.deviation = deviation
        self.last_archived_point = None

    def compress(self, new_point):
        # O primeiro ponto é sempre arquivado para ter uma referência.
        if self.last_archived_point is None:
            self.last_archived_point = new_point
            return [new_point]

        # Verifica a variação em relação ao último ponto que foi salvo.
        if abs(new_point[1] - self.last_archived_point[1]) >= self.deviation:
            self.last_archived_point = new_point
            return [new_point]
        
        # Se a variação for menor que o desvio, o ponto não é significativo.
        return []