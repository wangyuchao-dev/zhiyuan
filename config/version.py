"""版本号唯一来源。

为什么单独一个文件：main.py 的 OpenAPI 版本、镜像 tag、CHANGELOG 都要对得上，
散在多处迟早不一致。CI 里有一条断言校验它与 pyproject.toml 相同。
"""

__version__ = "1.3.0"
__version_info__ = tuple(int(p) for p in __version__.split(".") if p.isdigit())
