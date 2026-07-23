import json
import requests
import pandas as pd
from datetime import datetime
from itertools import product

url = "https://api.jiucaishuo.com/gz/gz/fed"

#股债风险溢价（FED 模型）的计算公式是 ERP = 1/指数 PE − 10 年期国债收益率，数值越高说明股票相对债券越便宜、性价比越高

# 你提供的完整参数（注意：直接发可能会被服务端拒绝，先看原始响应）
# payload = {
#     "category_type": "cz",
#     "gu_code": "881001.WI",
#     "year": 5,
#     "pe_category": "fed",
#     "year2": 1,
#     "mz": "",
#     "type": "h5",
#     "version": "2.5.9",
#     "ss": "",
#     "act_time": 1784723673559,
#     "tirgkjfs": "d4",
#     "abiokytke": "7f",
#     "u54rg5d": "4d",
#     "kf54ge7": "e",
#     "tiklsktr4": "4",
#     "lksytkjh": "e575",
#     "sbnoywr": "fd",
#     "bgd7h8tyu54": "d6",
#     "y654b5fs3tr": "7",
#     "bioduytlw": "2",
#     "bd4uy742": "8",
#     "h67456y": "1e5",
#     "bvytikwqjk": "d6",
#     "ngd4uy551": "e5",
#     "bgiuytkw": "68",
#     "nd354uy4752": "2",
#     "ghtoiutkmlg": "7ca",
#     "bd24y6421f": "d0",
#     "tbvdiuytk": "1",
#     "ibvytiqjek": "90",
#     "jnhf8u5231": "68",
#     "fjlkatj": "4d6",
#     "hy5641d321t": "08",
#     "iogojti": "0",
#     "ngd4yut78": "ca",
#     "nkjhrew": "8",
#     "yt447e13f": "6",
#     "n3bf4uj7y7": "5",
#     "nbf4uj7y432": "7f",
#     "yi854tew": "12",
#     "h13ey474": "12e",
#     "quikgdky": "d7"
# }

#year：3、5、10、15、20、-1 表示所有
#category_type:bz(比值)\cz(差值)
#pe_category:fed（fed 风险溢价）\xilv（股息率溢价）
#gu_code:881001.WI（万得全A）\HSI.HI（恒生指数）\000906.SH（中证800）、 000300.SH（沪深00）、000905.SH（中证500）、000016.SH（上证50）、000852.SH（中证1000）
pe_categories = ["fed","xilv"]
gu_codes = ["881001.WI","HSI.HI","000906.SH","000300.SH","000905.SH","000016.SH","000852.SH"]
category_types = ["bz","cz"]
# years = [3,5,10,-1]



def build_payload(category_type="cz",gu_code="881001.WI",year=5,pe_category="fed"):
    return {
        "category_type": category_type,
        "gu_code": gu_code,   # 万得全A
        "year": year,
        "pe_category": pe_category,
        "year2": 1,
        "type": "h5",
        "version": "2.5.9",
        "act_time":  int(datetime.now().timestamp())
    }

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Referer": "https://app.jiucaishuo.com/",
    "Accept": "application/json, text/plain, */*"
}

def get_fed_fengxian(category_type="cz",gu_code="881001.WI",year=5,pe_category="fed"):
    payload=build_payload(category_type,gu_code,year,pe_category)
    df = request_fed_fengxian(payload)
    if df is None:
        return None
    # 导出 CSV
    csv_name = f"mock_files/股债指数风险溢价_{category_type}_{gu_code}_{year}_{pe_category}.csv"
    df.to_csv(csv_name, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已导出 {len(df)} 条数据 -> {csv_name}")
    print(df.head())


def request_fed_fengxian(payload):
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    print(f"HTTP 状态码: {resp.status_code}")
    print(f"原始响应（前500字符）:\n{resp.text[:500]}\n")

    data = resp.json()

    # 如果被拒（code != 0），直接退出并提示看原始响应
    if data.get("code") != 0:
        print(f"[!] 接口返回错误：{data.get('message')}，请检查参数或抓包更新签名")
        exit()

    # 情况1：echarts 风格 {data: {tb_data: {x_data:[], series:[{name, data:[]}]}}}
    tb = data.get("data", {}).get("tb_data")
    if tb and "x_data" in tb and "series" in tb:
        dates = tb["x_data"]
        csv_data = {"date": dates}
        for idx, serie in enumerate(tb.get("series", [])):
            col_name = serie.get("name", f"指标_{idx}")
            data = serie.get("data", [])
            # 处理数据长度不一致的情况
            if len(data) != len(dates):
                # 如果数据较短，用 None 填充
                if len(data) < len(dates):
                    data = data + [None] * (len(dates) - len(data))
                # 如果数据较长，截断
                else:
                    data = data[:len(dates)]
            csv_data[col_name] = data
        df = pd.DataFrame(csv_data)
    # 情况2：列表风格 {data: {list: [{date, val}, ...]}}
    elif isinstance(data.get("data"), dict) and "list" in data["data"]:
        df = pd.DataFrame(data["data"]["list"])

    # 情况3：直接键值风格 {data: {date: [...], value: [...]}}
    elif isinstance(data.get("data"), dict):
        keys = [k for k in data["data"] if isinstance(data["data"][k], list)]
        print(keys)
        
        if len(keys) >= 2:
            df = pd.DataFrame({k: data["data"][k] for k in keys[:2]})
            df.columns = ["date", "risk_premium"] if len(df.columns) == 2 else df.columns
        else:
            print("[!] 无法自动识别返回结构，请手动查看 data 字段：")
            print(json.dumps(data["data"], ensure_ascii=False, indent=2)[:1000])
            exit()
    else:
        print("[!] 无法识别的 data 结构，原始响应：")
        print(json.dumps(data, ensure_ascii=False, indent=2)[:1000])
        exit()

    return df

if __name__ == "__main__":
    # 生成所有组合并调用函数
    for category_type, gu_code, pe_categorie in product(category_types, gu_codes, pe_categories):
        get_fed_fengxian(category_type, gu_code, 10, pe_categorie)
                

