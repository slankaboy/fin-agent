"""
announcement.py
巨潮资讯公告抓取 CLI 脚本。
核心逻辑已迁移至 fin_report_tools.CninfoReportProvider，
本文件保留 save / download / CLI 入口供独立使用。
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from fin_agent.tools.fin_report_tools import CninfoReportProvider, CATEGORY_MAP

load_dotenv()

_provider = CninfoReportProvider()


def fetch_cninfo_announcements(
    stock_code: str = None,
    start_date: str = None,
    end_date: str = None,
    category: str = "",
    page_size: int = 30,
    max_pages: int = 10,
    output_dir: str = None,
    delay: float = 1.0,
) -> pd.DataFrame:
    """从巨潮资讯网获取上市公司公告列表（委托 CninfoReportProvider）。"""
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
    today = datetime.today().strftime("%Y%m%d")
    return _provider._fetch(
        ts_code=stock_code or "",
        start_date=start_date or yesterday,
        end_date=end_date or today,
        category=category,
        max_pages=max_pages,
        page_size=page_size,
        delay=delay,
    )


def save_announcements(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    stock_code: str = None,
    output_dir: str = None,
) -> str:
    """将 DataFrame 保存为 CSV，返回文件路径。"""
    output_dir = output_dir or os.getenv("ANN_OUTPUT_DIR", "./data/cninfo")
    os.makedirs(output_dir, exist_ok=True)
    tag = f"_{stock_code}" if stock_code else ""
    file_name = f"{start_date}{tag}.csv" if start_date == end_date else f"{start_date}_{end_date}{tag}.csv"
    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"数据已保存: {file_path}  ({len(df)} 条)")
    return file_path


def download_announcements(
    df: pd.DataFrame,
    output_dir: str = None,
    delay: float = 0.5,
    overwrite: bool = False,
) -> pd.DataFrame:
    """批量下载公告 PDF（委托 CninfoReportProvider.download_pdfs）。"""
    return _provider.download_pdfs(df, output_dir=output_dir, delay=delay, overwrite=overwrite)


if __name__ == "__main__":
    import argparse

    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
    today = datetime.today().strftime("%Y%m%d")

    parser = argparse.ArgumentParser(description="从巨潮资讯网获取上市公司公告")
    parser.add_argument("--code", default=None, help="股票代码，如 000001 或 000001.SZ")
    parser.add_argument("--start", default=yesterday, help="开始日期 yyyymmdd，默认昨天")
    parser.add_argument("--end", default=today, help="结束日期 yyyymmdd，默认今天")
    parser.add_argument("--category", default="", help=f"公告类别，可选: {list(CATEGORY_MAP.keys())}")
    parser.add_argument("--pages", type=int, default=10, help="最多抓取页数，默认 10")
    parser.add_argument("--output", default=None, help="输出目录，默认读取 .env ANN_OUTPUT_DIR")
    parser.add_argument("--download", action="store_true", help="是否下载公告 PDF")
    parser.add_argument("--overwrite", action="store_true", help="下载时覆盖已存在文件")
    parser.add_argument("--csv", default=None, help="直接从已有 CSV 文件下载，跳过抓取")
    args = parser.parse_args()

    category_code = CATEGORY_MAP.get(args.category, args.category)

    if args.csv:
        df = pd.read_csv(args.csv)
        print(f"从 CSV 加载 {len(df)} 条记录: {args.csv}")
    else:
        df = fetch_cninfo_announcements(
            stock_code=args.code,
            start_date=args.start,
            end_date=args.end,
            category=category_code,
            max_pages=args.pages,
            output_dir=args.output,
        )
        if not df.empty:
            save_announcements(df, args.start, args.end, args.code, args.output)

    if not df.empty and args.download:
        df = download_announcements(df, output_dir=args.output, overwrite=args.overwrite)
        if not args.csv:
            save_announcements(df, args.start, args.end, args.code, args.output)

    if not df.empty:
        print(df.head().to_string())
