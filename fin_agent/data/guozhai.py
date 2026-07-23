import json
import requests
import pandas as pd
from datetime import datetime

url = "https://api.jiucaishuo.com/gz/gz/zhai"

payload = {
    "code": "numb",
    "year": 10,
    "type": "h5",
    "version": "2.5.9",
    "ss": "",
    "act_time": int(datetime.now().timestamp()),
    "tirgkjfs": "99",
    "abiokytke": "a2",
    "u54rg5d": "aa",
    "kf54ge7": "5",
    "tiklsktr4": "9",
    "lksytkjh": "f3b2",
    "sbnoywr": "8d",
    "bgd7h8tyu54": "7f",
    "y654b5fs3tr": "4",
    "bioduytlw": "0",
    "bd4uy742": "7",
    "h67456y": "1f3",
    "bvytikwqjk": "7f",
    "ngd4uy551": "f3",
    "bgiuytkw": "5b",
    "nd354uy4752": "8",
    "ghtoiutkmlg": "4f9",
    "bd24y6421f": "de",
    "tbvdiuytk": "1",
    "ibvytiqjek": "50",
    "jnhf8u5231": "5b",
    "fjlkatj": "aaa",
    "hy5641d321t": "e7",
    "iogojti": "e",
    "ngd4yut78": "f9",
    "nkjhrew": "7",
    "yt447e13f": "5",
    "n3bf4uj7y7": "3",
    "nbf4uj7y432": "a2",
    "yi854tew": "f8",
    "h13ey474": "f85",
    "quikgdky": "00"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Referer": "https://app.jiucaishuo.com/"
}

resp = requests.post(url, json=payload, headers=headers, timeout=15)
resp.raise_for_status()
data = resp.json()

# 校验返回状态
assert data.get("code") == 0, f"接口返回异常: {data}"
tb = data["data"]["tb_data"]

dates = tb["x_data"]
series = tb["series"][0]["data"]

# 组装 DataFrame
df = pd.DataFrame({"date": dates, "yield": series})
df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

print(f"标题: {data['data']['title']}")
print(f"最新值 ({data['data']['new']['date']}): {data['data']['new']['val']}")
print(f"数据条数: {len(df)}")
print(df.head())
print("...")

# 导出 CSV（utf-8-sig 保证 Excel 打开中文不乱码）
df.to_csv("mock_files/中国十年期国债收益率.csv", index=False, encoding="utf-8-sig")
print("已导出: mock_files/中国十年期国债收益率.csv")
