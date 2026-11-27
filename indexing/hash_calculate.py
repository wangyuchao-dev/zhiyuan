"""文件指纹：流式计算 MD5，作为「这份文档是否已入库」的唯一标识。

集合名直接用指纹，所以同一份文件重复上传不会重复建索引 —— 这是两层幂等的第一层。"""

import hashlib


class HashCalculate:
    """根据文件内容计算文件 hash"""

    def compute_hash_from_file(self, file_path: str) -> str:
        """从文件路径计算 MD5 哈希值（流式读取）"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
