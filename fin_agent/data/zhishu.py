
titles = [
            {
                "id": 6,
                "name": "自选",
                "types": 2
            },
            {
                "id": 8,
                "name": "精选",
                "types": 2
            },
            {
                "id": 2,
                "name": "规模",
                "types": 2
            },
            {
                "id": 1,
                "name": "行业",
                "types": 2
            },
            {
                "id": 9,
                "name": "策略",
                "types": 2
            },
            {
                "id": 3,
                "name": "主题",
                "types": 2
            },
            {
                "id": 5,
                "name": "境外",
                "types": 2
            },
            {
                "id": 7,
                "name": "其他",
                "types": 1
            }]



import requests
import pandas as pd
import json
import os
from datetime import datetime
import time

# 固定的请求参数（签名相关字段）
SIGN_PARAMS = {
    "tirgkjfs": "d8",
    "abiokytke": "3c",
    "u54rg5d": "fc",
    "kf54ge7": "8",
    "tiklsktr4": "8",
    "lksytkjh": "5d24",
    "sbnoywr": "e7",
    "bgd7h8tyu54": "77",
    "y654b5fs3tr": "f",
    "bioduytlw": "0",
    "bd4uy742": "d",
    "h67456y": "f5d",
    "bvytikwqjk": "77",
    "ngd4uy551": "5d",
    "bgiuytkw": "1c",
    "nd354uy4752": "d",
    "ghtoiutkmlg": "f31",
    "bd24y6421f": "78",
    "tbvdiuytk": "f",
    "ibvytiqjek": "ce",
    "jnhf8u5231": "1c",
    "fjlkatj": "fc1",
    "hy5641d321t": "8d",
    "iogojti": "8",
    "ngd4yut78": "31",
    "nkjhrew": "d",
    "yt447e13f": "c",
    "n3bf4uj7y7": "d",
    "nbf4uj7y432": "3c",
    "yi854tew": "6d",
    "h13ey474": "6d8",
    "quikgdky": "93"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.jiucaishuo.com/",
    "Origin": "https://www.jiucaishuo.com",
}

URL = "https://apiv2.jiucaishuo.com/indexvaluation/industry/group-list"


def fetch_page(category, types, page=1):
    """请求单页数据，返回 (records, update_time) 或 (None, None)"""
    current_timestamp = int(datetime.now().timestamp())
    payload = {
        "category": category,
        "types": str(types),
        "p": page,
        "field": "",
        "order_by": "",
        "type": "h5",
        "version": "2.5.9",
        "ss": "",
        "act_time": current_timestamp,
    }
    payload.update(SIGN_PARAMS)

    try:
        response = requests.post(URL, data=payload, headers=HEADERS, timeout=15)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ❌ 请求失败: {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON解析失败: {e}")
        return None, None

    data = result.get('data', {})
    if isinstance(data, dict):
        records = data.get('list', [])
        update_time = data.get('update_time', '')
    elif isinstance(data, list):
        records = data
        update_time = ''
    else:
        records = []
        update_time = ''

    return records, update_time


def flatten_records(records, update_time, category_name):
    """将嵌套数据扁平化为统一结构"""
    flat_records = []
    for item in records:
        flat_item = {
            'code': item.get('gu_code', ''),
            'name': item.get('gu_name', ''),
            'level': item.get('level', ''),
            'tag': item.get('tag', ''),
            'category': category_name,
            'update_time': update_time,
        }
        item_list = item.get('list', [])
        if len(item_list) >= 7:
            flat_item['pe'] = item_list[0].get('val', '')
            flat_item['pe_percent'] = item_list[1].get('val', '')
            flat_item['pb'] = item_list[2].get('val', '')
            flat_item['pb_percent'] = item_list[3].get('val', '')
            flat_item['dividend_yield'] = item_list[4].get('val', '')
            flat_item['dividend_percent'] = item_list[5].get('val', '')
            flat_item['roe'] = item_list[6].get('val', '')
        flat_records.append(flat_item)
    return flat_records


def fetch_category_data(category_id, types, category_name, max_pages=10):
    """获取某个分类的所有页数据"""
    all_flat = []
    update_time = ''
    for page in range(1, max_pages + 1):
        print(f"  获取第 {page} 页...")
        records, ut = fetch_page(category_id, types, page=page)
        if not records:
            print(f"  第 {page} 页无数据，停止翻页")
            break
        if not update_time and ut:
            update_time = ut
        flat = flatten_records(records, update_time, category_name)
        all_flat.extend(flat)
        print(f"  第 {page} 页: {len(records)} 条记录")
        # 请求间隔，避免频率限制
        time.sleep(0.5)
    return all_flat, update_time


def fetch_all_categories(output_dir="mock_files"):
    """迭代 titles 获取所有分类数据，分别保存为 CSV"""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    summary = []

    for title in titles:
        category_id = title['id']
        types = title['types']
        name = title['name']
        print(f"\n{'='*50}")
        print(f"正在获取分类: {name} (id={category_id}, types={types})")
        print(f"{'='*50}")

        flat_records, update_time = fetch_category_data(category_id, types, name,1)
        if not flat_records:
            print(f"⚠️ 分类 [{name}] 无数据，跳过")
            summary.append({'category': name, 'count': 0, 'file': ''})
            continue

        df = pd.DataFrame(flat_records)
        df['_fetch_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 文件名格式: 指数估值_name_date.csv
        csv_filename = os.path.join(output_dir, f"指数估值_{name}_{date_str}.csv")
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"✓ [{name}] 共 {len(df)} 条记录，已保存到: {csv_filename}")

        summary.append({'category': name, 'count': len(df), 'file': csv_filename})

    return summary


if __name__ == "__main__":
    print("开始获取韭财说所有分类指数估值数据...")
    print("-" * 50)

    summary = fetch_all_categories(output_dir="mock_files")

    print("\n" + "=" * 60)
    print("获取完成！汇总如下：")
    print("=" * 60)
    total = 0
    for s in summary:
        print(f"  {s['category']:<8} | {s['count']:>4} 条 | {s['file']}")
        total += s['count']
    print(f"\n总计: {total} 条记录，共 {len(summary)} 个分类")
    print("\n处理完成!")
