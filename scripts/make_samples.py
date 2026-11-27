"""生成 samples/ 下的边界样例 PDF（可重复执行，覆盖旧文件）。

    python scripts/make_samples.py

生成三份文件，分别对应三条容易被忽略的代码路径：

三份分别对应三条代码路径：
  04 带页眉页脚 → 清洗必须把它们去掉，页码 metadata 要保住
  05 扫描件     → 提取不到文字，入库必须拒收（400），不能灌空块进向量库
  06 长文档     → 多页长文，用来测上下文预算与跨页检索
"""

import os

import pymupdf

FONT = r"C:\Windows\Fonts\msyh.ttc"
OUT = r"C:\Users\ROG\Desktop\项目\知渊\AI_project源码\samples"

SECTIONS = [
    (
        "一、总则",
        "为规范公司采购与报销行为，明确审批权限与责任边界，依据《企业内部控制基本规范》制定本办法。"
        "本办法适用于公司总部及各分支机构全体员工。凡涉及采购申请、合同签订、验收付款、费用报销的，"
        "均须按本办法执行。各部门负责人对本部门提交的采购与报销事项的真实性负责。",
    ),
    (
        "二、采购申请",
        "单笔采购金额在人民币五千元以下的，由部门负责人审批；五千元至五万元的，须经分管副总审批；"
        "五万元以上的，须提交总经理办公会审议。采购申请须提前三个工作日提交，紧急采购须在事后两个工作日内补办手续。"
        "同一供应商同一品类的采购，不得为规避审批权限而拆分为多笔。",
    ),
    (
        "三、报销时限",
        "费用发生后须在三十日内完成报销，跨年度费用最迟不得晚于次年一月十五日。"
        "超过时限的报销须附情况说明并经财务负责人签字。差旅费报销须提供行程单与发票原件，"
        "住宿标准按城市分级执行：一线城市每晚不超过六百元，二线城市不超过四百元，其他城市不超过三百元。",
    ),
    (
        "四、发票要求",
        "报销凭证须为增值税发票或财政部监制的收据。发票抬头须与公司全称一致，税号须准确。"
        "发票内容须与实际业务相符，不得开具「办公用品」等笼统品名。电子发票须提供原件文件，"
        "打印件不作为报销凭证。发票遗失的，须取得开票方出具的证明并加盖公章。",
    ),
    (
        "五、违规处理",
        "虚报冒领、重复报销、拆分发票规避审批的，一经查实，追回全部款项并按公司奖惩制度处理；"
        "情节严重的，移送司法机关。审批人未尽审核义务造成损失的，承担相应管理责任。"
        "财务部门每季度抽查不低于百分之十的报销单据，抽查结果纳入部门考核。",
    ),
]


def make_header_footer_pdf(path: str, pages: int = 5):
    """每页都有重复页眉与页码 —— 用来验证清洗是否真的生效。"""
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)  # A4
        # 页眉：每页一字不差，清洗层应识别为跨页重复并整行删除
        page.insert_textbox(
            pymupdf.Rect(56, 30, 540, 52),
            "星辰科技股份有限公司 · 采购与报销管理办法",
            fontname="msyh",
            fontfile=FONT,
            fontsize=9,
            color=(0.4, 0.4, 0.4),
        )
        # 页码：应被页码规则删掉
        page.insert_textbox(
            pymupdf.Rect(56, 800, 540, 822),
            f"第 {i + 1} 页 共 {pages} 页",
            fontname="msyh",
            fontfile=FONT,
            fontsize=9,
            color=(0.4, 0.4, 0.4),
        )
        title, body = SECTIONS[i % len(SECTIONS)]
        page.insert_textbox(
            pymupdf.Rect(56, 70, 540, 100),
            title,
            fontname="msyhbd",
            fontfile=r"C:\Windows\Fonts\msyhbd.ttc",
            fontsize=14,
        )
        page.insert_textbox(
            pymupdf.Rect(56, 105, 540, 780),
            body * 2,
            fontname="msyh",
            fontfile=FONT,
            fontsize=10.5,
            lineheight=1.6,
        )
    doc.subset_fonts()
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def make_scanned_pdf(path: str):
    """把文字渲染成图片再放进 PDF —— 模拟扫描件：有内容，但没有文字层。"""
    src = pymupdf.open()
    page = src.new_page(width=595, height=842)
    page.insert_textbox(
        pymupdf.Rect(56, 60, 540, 780),
        SECTIONS[0][1] * 3,
        fontname="msyh",
        fontfile=FONT,
        fontsize=11,
        lineheight=1.6,
    )
    pix = page.get_pixmap(dpi=150)

    out = pymupdf.open()
    p2 = out.new_page(width=595, height=842)
    p2.insert_image(pymupdf.Rect(0, 0, 595, 842), pixmap=pix)
    out.subset_fonts()
    out.save(path, garbage=4, deflate=True)
    out.close()
    src.close()


def make_long_pdf(path: str, pages: int = 10):
    """长文档：每页一节，内容连续，用来测预算与跨页检索。"""
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            pymupdf.Rect(56, 30, 540, 52),
            "星辰科技股份有限公司 · 员工手册（2026 版）",
            fontname="msyh",
            fontfile=FONT,
            fontsize=9,
            color=(0.45, 0.45, 0.45),
        )
        page.insert_textbox(
            pymupdf.Rect(56, 800, 540, 822),
            f"第 {i + 1} 页 共 {pages} 页",
            fontname="msyh",
            fontfile=FONT,
            fontsize=9,
            color=(0.45, 0.45, 0.45),
        )
        title, body = SECTIONS[i % len(SECTIONS)]
        page.insert_textbox(
            pymupdf.Rect(56, 70, 540, 100),
            f"第{['一', '二', '三', '四', '五', '六', '七', '八', '九', '十'][i]}章  {title.split('、')[-1]}",
            fontname="msyhbd",
            fontfile=r"C:\Windows\Fonts\msyhbd.ttc",
            fontsize=15,
        )
        # 填充到接近整页，制造真实的长文档
        page.insert_textbox(
            pymupdf.Rect(56, 110, 540, 790),
            (body + "\n\n") * 3,
            fontname="msyh",
            fontfile=FONT,
            fontsize=10.5,
            lineheight=1.65,
        )
    doc.subset_fonts()
    doc.save(path, garbage=4, deflate=True)
    doc.close()


if __name__ == "__main__":
    make_header_footer_pdf(os.path.join(OUT, "04-带页眉页脚的采购报销办法.pdf"), pages=5)
    make_scanned_pdf(os.path.join(OUT, "05-扫描件无文字层.pdf"))
    make_long_pdf(os.path.join(OUT, "06-员工手册长文档.pdf"), pages=10)
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f"{os.path.getsize(p) / 1024:8.1f} KB  {f}")
