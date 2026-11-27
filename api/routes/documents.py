"""文档路由：上传 → 切分入库 → 绑定到会话，以及查询/解绑已绑定文档。

上传是离线动作（几秒到几十秒），所以和在线问答拆开成两个接口。"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.deps import get_agent, get_ingestor
from api.schemas import DocumentUploadResponse
from indexing.hash_calculate import HashCalculate

router = APIRouter(prefix="/api", tags=["documents"])

UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads"
)
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
        session_id: str = Form(...),
        file: UploadFile = File(...),
):
    """
    上传文档 → 切分入库 → 绑定到当前会话

    对应架构图里的「POST /api/documents，multipart 上传 PDF → 建索引」
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="目前只支持 PDF 格式")

    safe_name = os.path.basename(file.filename)
    dest = os.path.join(UPLOAD_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 算指纹 → 入库（已入过库的会跳过向量化）
        file_hash = HashCalculate().compute_hash_from_file(dest)
        get_ingestor().build_from_source(dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败：{e}")

    get_agent().bind_document(session_id, file_hash)
    bound = get_agent().get_bound_documents(session_id)

    return DocumentUploadResponse(
        file_hash=file_hash,
        filename=safe_name,
        bound_count=len(bound),
        message=f"上传成功，当前会话共 {len(bound)} 份文档，可以开始提问了",
    )


@router.get("/documents")
def list_documents(session_id: str):
    """
    查当前会话已绑定了哪些文档。

    多文档检索上线后，前端需要知道「这次会话到底传了几份」，
    否则用户不清楚自己的问题会被哪些文档覆盖。
    """
    bound = get_agent().get_bound_documents(session_id)
    return {"session_id": session_id, "count": len(bound), "file_hashes": bound}


@router.delete("/documents")
def unbind_documents(session_id: str):
    """解绑本会话的全部文档（保留对话记忆）"""
    get_agent().unbind_documents(session_id)
    return {"session_id": session_id, "count": 0, "file_hashes": []}
