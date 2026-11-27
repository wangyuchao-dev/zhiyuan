# 生产镜像：多阶段不必要（纯 Python），但要做三件事——
# 非 root 运行、依赖层单独缓存、健康检查交给编排系统。
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production

WORKDIR /app

# pymupdf 运行期需要这些系统库；先装再拷代码，改代码不会让这层失效
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 非 root：容器被拿下时，攻击者拿到的只是一个没有写权限的普通用户
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/runtime /app/indexing/chroma_db/uploads \
    && chown -R app:app /app/runtime /app/indexing/chroma_db /app/uploads
USER app

EXPOSE 8000

# 就绪检查用 liveness，不碰模型与数据库：依赖挂了也不该被反复重启
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
