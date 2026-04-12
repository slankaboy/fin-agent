"""
fin_report_tools.py
财务报表抽象层 + 多数据源实现

Provider 体系:
  FinReportProvider (ABC)
    ├── TushareReportProvider   — Tushare Pro API
    └── CninfoReportProvider    — 巨潮资讯网 (cninfo.com.cn)

对外接口（供 tushare_tools.py dispatcher 调用）:
  get_income_statement / get_balance_sheet / get_cash_flow /
  get_financial_indicator / get_announcements
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import tushare as ts

from fin_agent.config import Config

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _fmt_date_dash(d: str) -> str:
    """yyyymmdd → yyyy-mm-dd"""
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _default_dates(days_back: int = 730):
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    return start, end


# ── 抽象基类 ──────────────────────────────────────────────────────────────────

class FinReportProvider(ABC):
    """财务报表数据提供者抽象基类。"""

    @abstractmethod
    def get_income_statement(
        self, ts_code: str, period: str = None,
        start_date: str = None, end_date: str = None
    ) -> str:
        """利润表，返回 JSON 字符串。"""

    @abstractmethod
    def get_balance_sheet(
        self, ts_code: str, period: str = None,
        start_date: str = None, end_date: str = None
    ) -> str:
        """资产负债表，返回 JSON 字符串。"""

    @abstractmethod
    def get_cash_flow(
        self, ts_code: str, period: str = None,
        start_date: str = None, end_date: str = None
    ) -> str:
        """现金流量表，返回 JSON 字符串。"""

    @abstractmethod
    def get_financial_indicator(
        self, ts_code: str, period: str = None,
        start_date: str = None, end_date: str = None
    ) -> str:
        """核心财务指标，返回 JSON 字符串。"""

    def get_announcements(
        self, ts_code: str, start_date: str = None,
        end_date: str = None, category: str = ""
    ) -> str:
        """公告列表（可选实现），返回 JSON 字符串。"""
        return json.dumps({"error": f"{self.__class__.__name__} 不支持公告查询"}, ensure_ascii=False)


# ── Tushare 实现 ──────────────────────────────────────────────────────────────

class TushareReportProvider(FinReportProvider):
    """基于 Tushare Pro API 的财务报表实现。"""

    def _pro(self):
        ts.set_token(Config.TUSHARE_TOKEN)
        return ts.pro_api()

    def _query(self, api_fn, ts_code, period, start_date, end_date, fields, report_type="1"):
        kwargs = dict(ts_code=ts_code, fields=fields)
        if report_type:
            kwargs["report_type"] = report_type
        if period:
            kwargs["period"] = period
        else:
            if start_date:
                kwargs["start_date"] = start_date
            if end_date:
                kwargs["end_date"] = end_date
        df = api_fn(**kwargs)
        if df is None or df.empty:
            return None
        return df.sort_values("end_date", ascending=False).head(8)

    def get_income_statement(self, ts_code, period=None, start_date=None, end_date=None) -> str:
        try:
            df = self._query(
                self._pro().income, ts_code, period, start_date, end_date,
                fields=(
                    "ts_code,ann_date,f_ann_date,end_date,report_type,"
                    "total_revenue,revenue,total_cogs,"
                    "operate_profit,total_profit,income_tax,n_income,n_income_attr_p,"
                    "basic_eps,diluted_eps"
                )
            )
            return df.to_json(orient="records", force_ascii=False) if df is not None \
                else f"未找到 {ts_code} 的利润表数据。"
        except Exception as e:
            return f"获取利润表失败: {e}"

    def get_balance_sheet(self, ts_code, period=None, start_date=None, end_date=None) -> str:
        try:
            df = self._query(
                self._pro().balancesheet, ts_code, period, start_date, end_date,
                fields=(
                    "ts_code,ann_date,f_ann_date,end_date,report_type,"
                    "total_assets,total_liab,total_hldr_eqy_exc_min_int,"
                    "money_cap,accounts_receiv,inventories,"
                    "lt_borr,st_borr,notes_payable,accounts_payable"
                )
            )
            return df.to_json(orient="records", force_ascii=False) if df is not None \
                else f"未找到 {ts_code} 的资产负债表数据。"
        except Exception as e:
            return f"获取资产负债表失败: {e}"

    def get_cash_flow(self, ts_code, period=None, start_date=None, end_date=None) -> str:
        try:
            df = self._query(
                self._pro().cashflow, ts_code, period, start_date, end_date,
                fields=(
                    "ts_code,ann_date,f_ann_date,end_date,report_type,"
                    "net_profit,finan_exp,c_fr_sale_sg,"
                    "n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act,"
                    "free_cashflow,end_bal_cash"
                )
            )
            return df.to_json(orient="records", force_ascii=False) if df is not None \
                else f"未找到 {ts_code} 的现金流量表数据。"
        except Exception as e:
            return f"获取现金流量表失败: {e}"

    def get_financial_indicator(self, ts_code, period=None, start_date=None, end_date=None) -> str:
        try:
            df = self._query(
                self._pro().fina_indicator, ts_code, period, start_date, end_date,
                fields=(
                    "ts_code,ann_date,end_date,"
                    "eps,bps,roe,roa,gross_profit_margin,net_profit_margin,"
                    "debt_to_assets,current_ratio,quick_ratio,"
                    "inv_turn,ar_turn,assets_turn"
                ),
                report_type=None  # fina_indicator 无 report_type 参数
            )
            return df.to_json(orient="records", force_ascii=False) if df is not None \
                else f"未找到 {ts_code} 的财务指标数据。"
        except Exception as e:
            return f"获取财务指标失败: {e}"


# ── 巨潮资讯实现 ──────────────────────────────────────────────────────────────

_CNINFO_BASE_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
_CNINFO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

# 公告类别代码映射
CATEGORY_MAP = {
    "全部": "",
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
    "季报": "category_jdbg_szsh",
    "业绩预告": "category_yjyg_szsh",
    "业绩快报": "category_yjkb_szsh",
    "增发": "category_zf_szsh",
    "配股": "category_pg_szsh",
    "重大事项": "category_zdsx_szsh",
    "股权激励": "category_gqjl_szsh",
    "分红": "category_fh_szsh",
}


class CninfoReportProvider(FinReportProvider):
    """
    基于巨潮资讯网的财务报表实现。
    财报三表通过公告列表接口获取（按类别筛选），
    同时支持 get_announcements 通用公告查询。
    """

    def _stock_param(self, ts_code: str) -> str:
        """将 '000001.SZ' 转为巨潮格式 '000001,sz'。"""
        code = ts_code.split(".")[0]
        suffix_raw = ts_code.split(".")[-1].lower() if "." in ts_code else ""
        if suffix_raw in ("sz", "sh"):
            suffix = suffix_raw
        elif code.startswith(("0", "3")):
            suffix = "sz"
        else:
            suffix = "sh"
        return f"{code},{suffix}"

    def _fetch(
        self, ts_code: str, start_date: str, end_date: str,
        category: str = "", max_pages: int = 5, page_size: int = 30,
        delay: float = 0.8,
    ) -> pd.DataFrame:
        """核心抓取逻辑，返回 DataFrame。"""
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
        today = datetime.today().strftime("%Y%m%d")
        start_date = start_date or yesterday
        end_date = end_date or today

        stock_param = self._stock_param(ts_code) if ts_code else ""
        session = requests.Session()
        session.headers.update(_CNINFO_HEADERS)
        all_records = []

        for page in range(1, max_pages + 1):
            payload = {
                "stock": stock_param,
                "tabName": "fulltext",
                "pageSize": page_size,
                "pageNum": page,
                "column": "szse",
                "category": category,
                "plate": "",
                "seDate": f"{_fmt_date_dash(start_date)}~{_fmt_date_dash(end_date)}",
                "searchkey": "",
                "secid": "",
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            }
            try:
                resp = session.post(_CNINFO_BASE_URL, data=payload, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                break

            announcements = data.get("announcements") or []
            if not announcements:
                break

            for ann in announcements:
                ann["category"] = category
            all_records.extend(announcements)

            total = data.get("totalAnnouncement", 0)
            if page * page_size >= total:
                break
            time.sleep(delay)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        keep = [
            "secCode", "secName", "announcementTitle", "announcementTime",
            "category", "announcementTypeName", "adjunctUrl", "announcementId",
        ]
        df = df[[c for c in keep if c in df.columns]].copy()
        if "announcementTime" in df.columns:
            df["announcementTime"] = pd.to_datetime(
                df["announcementTime"], unit="ms", errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
        if "adjunctUrl" in df.columns:
            df["pdfUrl"] = df["adjunctUrl"].apply(
                lambda x: f"https://static.cninfo.com.cn/{x}" if pd.notna(x) and x else ""
            )
        return df

    def _period_to_dates(self, period: Optional[str], start_date: Optional[str], end_date: Optional[str]):
        """将 period/start_date/end_date 统一转为 (start, end) 字符串。"""
        if period:
            # 报告期当年全年范围
            year = period[:4]
            return f"{year}0101", f"{year}1231"
        today = datetime.today().strftime("%Y%m%d")
        two_years_ago = (datetime.today() - timedelta(days=730)).strftime("%Y%m%d")
        return start_date or two_years_ago, end_date or today

    def get_income_statement(self, ts_code, period=None, start_date=None, end_date=None) -> str:
        s, e = self._period_to_dates(period, start_date, end_date)
        df = self._fetch(ts_code, s, e, category=CATEGORY_MAP["年报"])
        if df.empty:
            # 降级：尝试季报
            df = self._fetch(ts_code, s, e, category=CATEGORY_MAP["季报"])
        if df.empty:
            return f"未从巨潮获取到 {ts_code} 的利润表相关公告。"
        return df.to_json(orient="records", force_ascii=False)

    def get_balance_sheet(self, ts_code, period=None, start_date=None, end_date=None) -> str:
        s, e = self._period_to_dates(period, start_date, end_date)
        df = self._fetch(ts_code, s, e, category=CATEGORY_MAP["年报"])
        if df.empty:
            return f"未从巨潮获取到 {ts_code} 的资产负债表相关公告。"
        return df.to_json(orient="records", force_ascii=False)

    def get_cash_flow(self, ts_code, period=None, start_date=None, end_date=None) -> str:
        s, e = self._period_to_dates(period, start_date, end_date)
        df = self._fetch(ts_code, s, e, category=CATEGORY_MAP["年报"])
        if df.empty:
            return f"未从巨潮获取到 {ts_code} 的现金流量表相关公告。"
        return df.to_json(orient="records", force_ascii=False)

    def get_financial_indicator(self, ts_code, period=None, start_date=None, end_date=None) -> str:
        s, e = self._period_to_dates(period, start_date, end_date)
        df = self._fetch(ts_code, s, e, category=CATEGORY_MAP["业绩快报"])
        if df.empty:
            df = self._fetch(ts_code, s, e, category=CATEGORY_MAP["业绩预告"])
        if df.empty:
            return f"未从巨潮获取到 {ts_code} 的财务指标相关公告。"
        return df.to_json(orient="records", force_ascii=False)

    def get_announcements(self, ts_code, start_date=None, end_date=None, category="") -> str:
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
        today = datetime.today().strftime("%Y%m%d")
        df = self._fetch(
            ts_code,
            start_date or yesterday,
            end_date or today,
            category=CATEGORY_MAP.get(category, category),
        )
        if df.empty:
            return f"未获取到 {ts_code} 的公告数据。"
        return df.to_json(orient="records", force_ascii=False)

    def download_pdfs(
        self, df: pd.DataFrame, output_dir: str = None,
        delay: float = 0.5, overwrite: bool = False,
    ) -> pd.DataFrame:
        """批量下载公告 PDF，返回带 localPath 列的 DataFrame。"""
        output_dir = output_dir or os.getenv("ANN_OUTPUT_DIR", "./data/cninfo")
        if "pdfUrl" not in df.columns or df.empty:
            return df

        session = requests.Session()
        session.headers.update({
            "User-Agent": _CNINFO_HEADERS["User-Agent"],
            "Referer": "https://www.cninfo.com.cn/",
        })
        local_paths = []

        for i, row in df.iterrows():
            pdf_url = row.get("pdfUrl", "")
            if not pdf_url:
                local_paths.append("")
                continue

            sec_code = str(row.get("secCode", "unknown"))
            sec_dir = os.path.join(output_dir, sec_code)
            os.makedirs(sec_dir, exist_ok=True)

            ann_time = str(row.get("announcementTime", ""))[:10].replace("-", "")
            original = pdf_url.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
            file_path = os.path.join(sec_dir, f"{sec_code}_{ann_time}_{original}.pdf")

            if os.path.exists(file_path) and not overwrite:
                local_paths.append(file_path)
                continue

            try:
                resp = session.get(pdf_url, timeout=30, stream=True)
                resp.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                local_paths.append(file_path)
            except requests.RequestException:
                local_paths.append("")
            time.sleep(delay)

        df = df.copy()
        df["localPath"] = local_paths
        return df


# ── 工厂 ──────────────────────────────────────────────────────────────────────

class FinReportFactory:
    """
    根据 provider 名称返回对应实现。
    支持: "tushare"（默认）、"cninfo"
    """
    _instances: dict[str, FinReportProvider] = {}

    @classmethod
    def get(cls, provider: str = "tushare") -> FinReportProvider:
        if provider not in cls._instances:
            if provider == "cninfo":
                cls._instances[provider] = CninfoReportProvider()
            else:
                cls._instances[provider] = TushareReportProvider()
        return cls._instances[provider]


def _provider(kwargs: dict) -> tuple[FinReportProvider, dict]:
    """从 kwargs 中提取 provider 参数，返回 (provider实例, 剩余kwargs)。"""
    provider_name = kwargs.pop("provider", "tushare")
    return FinReportFactory.get(provider_name), kwargs


# ── 对外函数（供 dispatcher 调用）────────────────────────────────────────────

def get_income_statement(ts_code, period=None, start_date=None, end_date=None, provider="tushare") -> str:
    return FinReportFactory.get(provider).get_income_statement(ts_code, period, start_date, end_date)

def get_balance_sheet(ts_code, period=None, start_date=None, end_date=None, provider="tushare") -> str:
    return FinReportFactory.get(provider).get_balance_sheet(ts_code, period, start_date, end_date)

def get_cash_flow(ts_code, period=None, start_date=None, end_date=None, provider="tushare") -> str:
    return FinReportFactory.get(provider).get_cash_flow(ts_code, period, start_date, end_date)

def get_financial_indicator(ts_code, period=None, start_date=None, end_date=None, provider="tushare") -> str:
    return FinReportFactory.get(provider).get_financial_indicator(ts_code, period, start_date, end_date)

def get_announcements(ts_code, start_date=None, end_date=None, category="", provider="cninfo") -> str:
    return FinReportFactory.get(provider).get_announcements(ts_code, start_date, end_date, category)


def download_financial_report(
    ts_code: str,
    category: str = "年报",
    start_date: str = None,
    end_date: str = None,
    max_count: int = 5,
    overwrite: bool = False,
) -> str:
    """
    从巨潮资讯下载财务报告 PDF 到本地 reports/ 目录。
    1. 先获取公告列表
    2. 取前 max_count 条有 PDF 的公告
    3. 下载到 fin_agent/reports/<ts_code>/ 目录
    返回下载结果摘要（JSON）。
    """
    import json as _json

    provider = FinReportFactory.get("cninfo")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    two_years_ago = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")

    category_code = CATEGORY_MAP.get(category, category)
    result_json = provider.get_announcements(
        ts_code,
        start_date=start_date or two_years_ago,
        end_date=end_date or yesterday,
        category=category_code,
    )

    try:
        records = _json.loads(result_json)
    except Exception:
        return result_json  # 直接返回错误信息

    if not records:
        return f"未找到 {ts_code} 的 {category} 公告，无法下载。"

    df = pd.DataFrame(records)

    # 只保留有 PDF 的条目，取前 max_count 条
    if "pdfUrl" in df.columns:
        df = df[df["pdfUrl"].notna() & (df["pdfUrl"] != "")].head(max_count)
    else:
        return f"公告列表中无 PDF 链接，无法下载。"

    if df.empty:
        return f"未找到带 PDF 的 {category} 公告。"

    # 下载目录：项目内 fin_agent/reports/<ts_code>/
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    output_dir = os.path.join(base_dir, ts_code.replace(".", "_"))

    df = provider.download_pdfs(df, output_dir=output_dir, overwrite=overwrite)

    # 汇总结果
    summary = []
    for _, row in df.iterrows():
        local = row.get("localPath", "")
        summary.append({
            "title": row.get("announcementTitle", ""),
            "date": row.get("announcementTime", ""),
            "localPath": local,
            "status": "成功" if local else "失败",
        })

    return _json.dumps(summary, ensure_ascii=False)


# ── Tool Schema ───────────────────────────────────────────────────────────────

_PROVIDER_PROP = {
    "provider": {
        "type": "string",
        "enum": ["tushare", "cninfo"],
        "description": "数据源：'tushare'（默认，结构化财务数据）或 'cninfo'（巨潮资讯，公告原文）。"
    }
}

FIN_REPORT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": (
                "获取A股上市公司利润表，包含营业收入、营业成本、营业利润、净利润、EPS等核心损益数据。"
                "默认使用 tushare 返回结构化数据（最近8期）；指定 provider='cninfo' 则返回巨潮公告列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {"type": "string", "description": "股票代码，如 '000001.SZ'。"},
                    "period": {"type": "string", "description": "报告期，格式 YYYYMMDD，如 '20231231'。"},
                    "start_date": {"type": "string", "description": "公告开始日期，格式 YYYYMMDD。"},
                    "end_date": {"type": "string", "description": "公告结束日期，格式 YYYYMMDD。"},
                    **_PROVIDER_PROP,
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": (
                "获取A股上市公司资产负债表，包含总资产、总负债、股东权益、货币资金、应收账款、存货等。"
                "默认使用 tushare；指定 provider='cninfo' 则返回巨潮年报公告列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {"type": "string", "description": "股票代码，如 '000001.SZ'。"},
                    "period": {"type": "string", "description": "报告期，格式 YYYYMMDD。"},
                    "start_date": {"type": "string", "description": "公告开始日期，格式 YYYYMMDD。"},
                    "end_date": {"type": "string", "description": "公告结束日期，格式 YYYYMMDD。"},
                    **_PROVIDER_PROP,
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cash_flow",
            "description": (
                "获取A股上市公司现金流量表，包含经营/投资/筹资活动现金流、自由现金流等。"
                "默认使用 tushare；指定 provider='cninfo' 则返回巨潮年报公告列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {"type": "string", "description": "股票代码，如 '000001.SZ'。"},
                    "period": {"type": "string", "description": "报告期，格式 YYYYMMDD。"},
                    "start_date": {"type": "string", "description": "公告开始日期，格式 YYYYMMDD。"},
                    "end_date": {"type": "string", "description": "公告结束日期，格式 YYYYMMDD。"},
                    **_PROVIDER_PROP,
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_indicator",
            "description": (
                "获取A股上市公司核心财务指标：ROE、ROA、毛利率、净利率、资产负债率、流动比率等。"
                "默认使用 tushare；指定 provider='cninfo' 则返回巨潮业绩快报/预告公告列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {"type": "string", "description": "股票代码，如 '000001.SZ'。"},
                    "period": {"type": "string", "description": "报告期，格式 YYYYMMDD。"},
                    "start_date": {"type": "string", "description": "公告开始日期，格式 YYYYMMDD。"},
                    "end_date": {"type": "string", "description": "公告结束日期，格式 YYYYMMDD。"},
                    **_PROVIDER_PROP,
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_announcements",
            "description": (
                "从巨潮资讯网获取上市公司公告列表，支持按类别筛选（年报、半年报、季报、业绩预告、"
                "重大事项、分红等）。返回公告标题、时间、PDF 下载链接等信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {"type": "string", "description": "股票代码，如 '000001.SZ'。"},
                    "start_date": {"type": "string", "description": "开始日期，格式 YYYYMMDD，默认昨天。"},
                    "end_date": {"type": "string", "description": "结束日期，格式 YYYYMMDD，默认今天。"},
                    "category": {
                        "type": "string",
                        "description": (
                            "公告类别，可选：全部、年报、半年报、季报、业绩预告、业绩快报、"
                            "增发、配股、重大事项、股权激励、分红。默认全部。"
                        )
                    },
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_financial_report",
            "description": (
                "从巨潮资讯网下载上市公司财务报告 PDF 到本地 reports/ 目录。"
                "支持年报、半年报、季报等类别，下载后返回本地文件路径列表。"
                "适合用户需要获取原始报告文件时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {"type": "string", "description": "股票代码，如 '000001.SZ'。"},
                    "category": {
                        "type": "string",
                        "description": "报告类别，可选：年报、半年报、季报、业绩预告、业绩快报。默认年报。"
                    },
                    "start_date": {"type": "string", "description": "公告开始日期，格式 YYYYMMDD，默认两年前。"},
                    "end_date": {"type": "string", "description": "公告结束日期，格式 YYYYMMDD，默认昨天。"},
                    "max_count": {
                        "type": "integer",
                        "description": "最多下载条数，默认 5。"
                    },
                },
                "required": ["ts_code"]
            }
        }
    },
]


def execute_fin_report_tool(tool_name: str, arguments: dict) -> str:
    """Dispatcher for fin_report tools."""
    fn_map = {
        "get_income_statement": get_income_statement,
        "get_balance_sheet": get_balance_sheet,
        "get_cash_flow": get_cash_flow,
        "get_financial_indicator": get_financial_indicator,
        "get_announcements": get_announcements,
        "download_financial_report": download_financial_report,
    }
    fn = fn_map.get(tool_name)
    if fn is None:
        return None
    return fn(**arguments)
